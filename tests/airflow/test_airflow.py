"""Tests for the `AirflowDurability` capability.

Behavior is driven through `Agent(..., capabilities=[AirflowDurability(storage=...)])` against an
in-memory store, so there is no Airflow, Docker, or network dependency. The two production behaviors
the capability relies on -- positional step keying and a retry that serves stored entries without
re-running -- are reproduced by re-running the agent against the populated store.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    BinaryContent,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolReturn,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests, ToolDefinition
from pydantic_ai.toolsets import ExternalToolset, FunctionToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset  # pyright: ignore[reportPrivateUsage]
from pydantic_core import PydanticSerializationError

from pydantic_ai_harness.airflow import (
    DURABLE_KEY_PREFIX,
    AirflowDurability,
    AirflowDurabilityWarning,
    InMemoryDurableStorage,
    JSONFileDurableStorage,
    StoredEntry,
)
from pydantic_ai_harness.airflow._capability import (  # pyright: ignore[reportPrivateUsage]
    _Skip,
    _step_counter,
    _StepCounter,
    _try_dump,
)
from pydantic_ai_harness.airflow._fingerprint import (  # pyright: ignore[reportPrivateUsage]
    _content_settings,
    _strip_message,
    fingerprint_model_request,
    fingerprint_token,
    fingerprint_tool_call,
)

from .conftest import RunLog, tool_then_text


def build_agent(storage: InMemoryDurableStorage, *, log: RunLog | None = None, **kwargs: Any) -> Agent[Any, Any]:
    """An agent with one `act` tool, memoized into `storage`."""
    agent = Agent(tool_then_text(log), name='a', capabilities=[AirflowDurability(storage=storage, **kwargs)])
    return agent


class TestMemoization:
    def test_model_and_tool_are_memoized(self) -> None:
        store = InMemoryDurableStorage()
        agent = build_agent(store)

        @agent.tool_plain
        def act() -> str:
            return 'sunny'

        result = agent.run_sync('go')

        assert result.output == 'done'
        assert store.keys == [
            f'{DURABLE_KEY_PREFIX}model_step_0',
            f'{DURABLE_KEY_PREFIX}tool_step_1',
            f'{DURABLE_KEY_PREFIX}model_step_2',
        ]

    def test_retry_replays_with_zero_re_execution(self) -> None:
        store = InMemoryDurableStorage()
        log = RunLog()

        def make() -> Agent[Any, Any]:
            agent = build_agent(store, log=log)

            @agent.tool_plain
            def act() -> str:
                log.tool_calls.append('act')
                return 'sunny'

            return agent

        first = make().run_sync('go')
        assert first.output == 'done'
        assert log.model_requests == 2
        assert log.tool_calls == ['act']

        # Simulate an Airflow task retry: a fresh agent against the populated store.
        second = make().run_sync('go')
        assert second.output == 'done'
        # Nothing re-executed: no new model requests, no new tool calls.
        assert log.model_requests == 2
        assert log.tool_calls == ['act']

    def test_crash_mid_run_reuses_completed_model_step_and_reruns_failed_tool(self) -> None:
        store = InMemoryDurableStorage()
        log = RunLog()
        attempts: list[int] = []

        def make() -> Agent[Any, Any]:
            agent = build_agent(store, log=log)

            @agent.tool_plain
            def act() -> str:
                attempts.append(1)
                if len(attempts) == 1:
                    raise RuntimeError('transient')
                log.tool_calls.append('act')
                return 'sunny'

            return agent

        with pytest.raises(RuntimeError, match='transient'):
            make().run_sync('go')
        # The model step completed and was stored; the tool failed before storing.
        assert store.keys == [f'{DURABLE_KEY_PREFIX}model_step_0']
        assert log.model_requests == 1

        result = make().run_sync('go')
        assert result.output == 'done'
        # The completed model step replayed; only the second model request ran live.
        assert log.model_requests == 2
        assert len(attempts) == 2


class TestFingerprintInvalidation:
    def test_changed_prompt_misses_and_warns(self) -> None:
        store = InMemoryDurableStorage()
        log = RunLog()

        def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            log.model_requests += 1
            return ModelResponse(parts=[TextPart('answered')])

        def make() -> Agent[Any, Any]:
            return Agent(FunctionModel(answer), name='a', capabilities=[AirflowDurability(storage=store)])

        make().run_sync('go')
        assert log.model_requests == 1

        with pytest.warns(AirflowDurabilityWarning, match='does not match the current request'):
            make().run_sync('a different prompt')
        # The stored response did not match the new request, so the model ran again.
        assert log.model_requests == 2

    def test_cascade_invalidation_via_tool_call_id(self) -> None:
        # A model divergence regenerates the tool_call_id, which is part of the tool fingerprint, so
        # a stale tool result no longer matches even at the same position with the same name/args.
        store = InMemoryDurableStorage()
        tool_runs: list[int] = []

        def make() -> Agent[Any, Any]:
            agent = Agent(tool_then_text(), name='a', capabilities=[AirflowDurability(storage=store)])

            @agent.tool_plain
            def act() -> str:
                tool_runs.append(1)
                return 'sunny'

            return agent

        make().run_sync('go')
        assert len(tool_runs) == 1

        # A new prompt forces a fresh model response with a new tool_call_id, cascading to the tool.
        with pytest.warns(AirflowDurabilityWarning):
            make().run_sync('other')
        assert len(tool_runs) == 2

    def test_run_time_model_id_folds_into_the_step_key(self) -> None:
        extra = TestModel(custom_output_text='from extra')
        store = InMemoryDurableStorage()
        agent = Agent(
            tool_then_text(), name='a', capabilities=[AirflowDurability(storage=store, models={'extra': extra})]
        )

        agent.run_sync('go', model='extra')

        assert f'{DURABLE_KEY_PREFIX}model_step_0.extra' in store.keys

    def test_string_default_model_keeps_a_suffixless_key(self) -> None:
        store = InMemoryDurableStorage()
        agent = Agent('test', name='a', capabilities=[AirflowDurability(storage=store)])

        agent.run_sync('go')

        assert store.keys[0] == f'{DURABLE_KEY_PREFIX}model_step_0'


class TestStructuredResults:
    """The real Airflow provider never memoizes these (its `json.dumps` skips them); this one does."""

    def _round_trips(self, tool_value: Any) -> None:
        store = InMemoryDurableStorage()
        log = RunLog()

        def make() -> Agent[Any, Any]:
            agent = build_agent(store, log=log)

            @agent.tool_plain
            def act() -> Any:
                log.tool_calls.append('act')
                return tool_value

            return agent

        assert make().run_sync('go').output == 'done'
        # Replay: the structured result is served from the store, tool not re-run.
        assert make().run_sync('go').output == 'done'
        assert log.tool_calls == ['act']
        assert f'{DURABLE_KEY_PREFIX}tool_step_1' in store.keys

    def test_tool_return_round_trips(self) -> None:
        self._round_trips(ToolReturn(return_value='ok', content='extra'))

    def test_binary_content_round_trips(self) -> None:
        self._round_trips(BinaryContent(data=b'\x89PNG', media_type='image/png'))


class TestControlFlowSignals:
    """Control-flow signals cross the store as values, and re-raise on replay without corrupting it."""

    def _run_with_signal(self, exc: type[Exception], store: InMemoryDurableStorage) -> Any:
        agent = Agent(
            tool_then_text(),
            name='a',
            output_type=[str, DeferredToolRequests],
            capabilities=[AirflowDurability(storage=store)],
        )

        @agent.tool_plain
        def act() -> str:
            raise exc

        return agent.run_sync('go').output

    def test_approval_required_is_memoized_and_replays(self) -> None:
        store = InMemoryDurableStorage()
        first = self._run_with_signal(ApprovalRequired, store)
        assert isinstance(first, DeferredToolRequests)
        assert len(first.approvals) == 1
        # Replay re-raises the stored signal rather than a corrupted value.
        second = self._run_with_signal(ApprovalRequired, store)
        assert isinstance(second, DeferredToolRequests)
        assert len(second.approvals) == 1

    def test_call_deferred_pauses_the_run(self) -> None:
        store = InMemoryDurableStorage()
        output = self._run_with_signal(CallDeferred, store)
        assert isinstance(output, DeferredToolRequests)
        assert len(output.calls) == 1

    def test_model_retry_is_not_a_store_failure(self) -> None:
        store = InMemoryDurableStorage()
        calls: list[int] = []
        agent = Agent(tool_then_text(), name='a', capabilities=[AirflowDurability(storage=store)])

        @agent.tool_plain
        def act() -> str:
            calls.append(1)
            if len(calls) == 1:
                raise ModelRetry('again')
            return 'ok'

        assert agent.run_sync('go').output == 'done'


class TestTransparency:
    def test_durable_false_is_transparent(self) -> None:
        store = InMemoryDurableStorage()
        agent = build_agent(store, durable=False)

        @agent.tool_plain
        def act() -> str:
            return 'sunny'

        result = agent.run_sync('go')

        assert result.output == 'done'
        assert store.keys == []

    def test_capability_absent_behaves_normally(self) -> None:
        agent = Agent(tool_then_text(), name='a')

        @agent.tool_plain
        def act() -> str:
            return 'sunny'

        assert agent.run_sync('go').output == 'done'


class TestNonSerializableResult:
    """A value that cannot be serialized is skipped with a warning, not a crash.

    In a full agent run Pydantic AI enforces serializable tool returns upstream (it puts them in the
    message history), so this safety net is exercised at the memoization boundary directly.
    """

    async def test_memoize_skips_a_non_serializable_value(self) -> None:
        store = InMemoryDurableStorage()
        cap: AirflowDurability[None] = AirflowDurability(storage=store)

        def dump_that_fails(_: object) -> Any:
            raise PydanticSerializationError('nope')

        token = _step_counter.set(_StepCounter())
        try:
            with pytest.warns(AirflowDurabilityWarning, match='not serializable'):
                result = await cap._memoize(
                    kind='tool',
                    suffix='',
                    fingerprint='fp',
                    run=_return_sentinel,
                    dump=dump_that_fails,
                    load=lambda payload: payload,
                )
        finally:
            _step_counter.reset(token)

        assert result == 'sentinel'
        assert store.keys == []

    def test_try_dump_returns_skip_on_serialization_error(self) -> None:
        def dump_that_fails(_: object) -> Any:
            raise PydanticSerializationError('nope')

        assert isinstance(_try_dump(object(), dump_that_fails), _Skip)

    def test_try_dump_returns_payload_on_success(self) -> None:
        payload: JsonValue = {'a': 1}
        assert _try_dump(payload, lambda value: value) == {'a': 1}


async def _return_sentinel() -> str:
    return 'sentinel'


class TestRuntimeToolsets:
    @pytest.mark.parametrize('kind', ['function', 'mcp', 'dynamic'])
    def test_executing_toolsets_added_per_run_are_rejected(self, kind: str) -> None:
        toolset: Any
        if kind == 'function':
            toolset = FunctionToolset[object](id='late')
        elif kind == 'dynamic':
            toolset = DynamicToolset[object](lambda ctx: FunctionToolset[object](), id='late')
        else:
            pytest.importorskip('pydantic_ai.mcp')
            from .test_airflow_mcp import FakeMCPToolset

            toolset = FakeMCPToolset(id='late')

        store = InMemoryDurableStorage()
        agent = build_agent(store)

        # The runtime toolset is rejected during run setup, before the model or any tool runs.
        with pytest.raises(UserError, match=re.escape('cannot be passed to `run(toolsets=...)` at runtime')):
            agent.run_sync('go', toolsets=[toolset])

    def test_non_executing_runtime_toolsets_pass_through(self) -> None:
        store = InMemoryDurableStorage()
        agent = build_agent(store)

        @agent.tool_plain
        def act() -> str:
            return 'sunny'

        result = agent.run_sync('go', toolsets=[ExternalToolset[object]([ToolDefinition(name='remote')], id='ext')])

        assert result.output == 'done'


class TestEnqueueGuard:
    def test_enqueue_inside_a_memoized_tool_raises(self) -> None:
        store = InMemoryDurableStorage()
        agent = Agent(tool_then_text(), name='a', capabilities=[AirflowDurability(storage=store)])

        @agent.tool
        def act(ctx: RunContext[object]) -> None:
            ctx.enqueue('later')

        with pytest.raises(UserError, match='enqueue'):
            agent.run_sync('go')

    def test_enqueue_from_a_model_event_handler_raises(self) -> None:
        store = InMemoryDurableStorage()

        async def handler(ctx: RunContext[object], stream: Any) -> None:
            # Enqueue on entry (before consuming the stream), so the model-unit guard fires directly.
            ctx.enqueue('later')

        agent = Agent(
            TestModel(), name='a', capabilities=[AirflowDurability(storage=store, event_stream_handler=handler)]
        )

        with pytest.raises(UserError, match='enqueue'):
            agent.run_sync('go')

    def test_enqueue_from_an_agent_event_handler_raises(self) -> None:
        store = InMemoryDurableStorage()

        from pydantic_ai.messages import FunctionToolCallEvent

        async def handler(ctx: RunContext[object], stream: Any) -> None:
            async for event in stream:
                if isinstance(event, FunctionToolCallEvent):
                    ctx.enqueue('later')

        agent = Agent(
            TestModel(), name='a', capabilities=[AirflowDurability(storage=store, event_stream_handler=handler)]
        )
        # A tool so the run emits a `FunctionToolCallEvent`; the guard fires before the tool runs.
        agent.tool_plain(act)

        with pytest.raises(UserError, match='enqueue'):
            agent.run_sync('go')


class TestPerToolConfig:
    def test_metadata_false_runs_the_tool_inline(self) -> None:
        store = InMemoryDurableStorage()
        toolset = FunctionToolset[object](id='tools')

        @toolset.tool_plain(metadata={'airflow': False})
        def act() -> str:
            return 'sunny'

        agent = Agent(tool_then_text(), name='a', toolsets=[toolset], capabilities=[AirflowDurability(storage=store)])

        agent.run_sync('go')

        # No tool step was stored: the tool ran inline.
        assert f'{DURABLE_KEY_PREFIX}tool_step_1' not in store.keys

    def test_non_empty_config_is_rejected(self) -> None:
        store = InMemoryDurableStorage()
        toolset = FunctionToolset[object](id='tools')
        toolset.add_function(act, metadata={'airflow': {'retries': 3}})
        agent = Agent(tool_then_text(), name='a', toolsets=[toolset], capabilities=[AirflowDurability(storage=store)])

        with pytest.raises(UserError, match='takes no per-tool options'):
            agent.run_sync('go')


def act() -> str:
    return 'sunny'


class TestBinding:
    def test_agent_without_a_name_is_rejected(self) -> None:
        with pytest.raises(UserError, match='unique `name`'):
            Agent(TestModel(), capabilities=[AirflowDurability(storage=InMemoryDurableStorage())])

    def test_capability_name_overrides_the_agent_name(self) -> None:
        store = InMemoryDurableStorage()
        agent = Agent(tool_then_text(), name='a', capabilities=[AirflowDurability(storage=store, name='custom')])
        agent.tool_plain(act)

        agent.run_sync('go')

        assert store.keys[0] == f'{DURABLE_KEY_PREFIX}model_step_0'
        cap = AirflowDurability.from_agent(agent)
        assert cap is not None
        assert cap.name == 'custom'

    def test_default_storage_is_in_memory(self) -> None:
        cap: AirflowDurability[None] = AirflowDurability()
        assert isinstance(cap._storage, InMemoryDurableStorage)  # pyright: ignore[reportPrivateUsage]

    def test_in_durable_context_reflects_the_flag(self) -> None:
        assert AirflowDurability(durable=True).in_durable_context is True
        assert AirflowDurability(durable=False).in_durable_context is False


class TestEventStreamHandler:
    def test_agent_events_are_memoized_and_replay(self) -> None:
        seen: list[AgentStreamEvent] = []

        async def handler(ctx: RunContext[object], stream: Any) -> None:
            async for event in stream:
                seen.append(event)

        def make() -> Agent[Any, Any]:
            agent = Agent(
                TestModel(),
                name='a',
                capabilities=[AirflowDurability(storage=store, event_stream_handler=handler)],
            )
            agent.tool_plain(act)
            return agent

        store = InMemoryDurableStorage()
        result = make().run_sync('go')
        assert result.output
        assert seen
        assert any(k.startswith(f'{DURABLE_KEY_PREFIX}model_stream_step') for k in store.keys)
        assert any(k.startswith(f'{DURABLE_KEY_PREFIX}event_stream_handler_step') for k in store.keys)

        seen.clear()
        replayed = make().run_sync('go')
        # The run completes from the store, and the handler's side effects are not repeated: model
        # events were delivered live on the first run, and agent-event deliveries are memoized.
        assert replayed.output == result.output
        assert seen == []


class TestCancelSuspendedResponse:
    def test_cancel_suspended_response_is_memoized(self) -> None:
        cancelled: list[ModelResponse] = []

        class CancellableModel(FunctionModel):
            async def cancel_suspended_response(self, response: ModelResponse) -> None:
                cancelled.append(response)

        def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not any(getattr(m, 'state', None) == 'suspended' for m in messages):
                return ModelResponse(parts=[TextPart(content='partial')], state='suspended')
            raise RuntimeError('continuation failed')

        store = InMemoryDurableStorage()
        agent = Agent(CancellableModel(fn, model_name='fn'), name='a', capabilities=[AirflowDurability(storage=store)])

        with pytest.raises(RuntimeError, match='continuation failed'):
            agent.run_sync('go')

        assert len(cancelled) == 1
        assert any(k.startswith(f'{DURABLE_KEY_PREFIX}model_cancel_step') for k in store.keys)


class TestDeferredToolHonesty:
    """Deferred tools genuinely cannot suspend-and-resume in this model.

    An `ApprovalRequired`/`CallDeferred` ends the run with a `DeferredToolRequests` output, memoized
    like any other tool result. There is no run-suspension: resolving the request is a fresh run that
    the caller drives, exactly as without durability. This test pins that current behavior.
    """

    def test_deferred_request_ends_the_run_without_suspension(self) -> None:
        store = InMemoryDurableStorage()
        agent = Agent(
            tool_then_text(),
            name='a',
            output_type=[str, DeferredToolRequests],
            capabilities=[AirflowDurability(storage=store)],
        )

        @agent.tool_plain
        def act() -> str:
            raise CallDeferred

        result = agent.run_sync('go')

        assert isinstance(result.output, DeferredToolRequests)
        # The run completed (it did not suspend); the deferred call was memoized as a value.
        assert any(k.startswith(f'{DURABLE_KEY_PREFIX}tool_step') for k in store.keys)


class TestStorageBackends:
    def test_in_memory_round_trips_and_cleans_up(self) -> None:
        store = InMemoryDurableStorage()
        assert store.load('missing') is None
        store.save('k', {'v': 1}, fingerprint='fp')
        entry = store.load('k')
        assert entry == StoredEntry(value={'v': 1}, fingerprint='fp')
        assert store.keys == ['k']
        store.cleanup()
        assert store.load('k') is None

    def test_json_file_round_trips_and_cleans_up(self, tmp_path: Path) -> None:
        path = tmp_path / 'nested' / 'cache.json'
        store = JSONFileDurableStorage(path)
        assert store.load('missing') is None  # file does not exist yet

        store.save('k', {'v': 1}, fingerprint='fp')
        assert store.load('k') == StoredEntry(value={'v': 1}, fingerprint='fp')

        store.save('k2', [1, 2], fingerprint=None)
        assert store.load('k2') == StoredEntry(value=[1, 2], fingerprint=None)

        store.cleanup()
        assert store.load('k') is None

    def test_json_file_ignores_malformed_content(self, tmp_path: Path) -> None:
        path = tmp_path / 'cache.json'
        path.write_text('not json{')
        store = JSONFileDurableStorage(path)
        assert store.load('k') is None

    def test_json_file_coerces_a_non_string_fingerprint_to_none(self, tmp_path: Path) -> None:
        path = tmp_path / 'cache.json'
        path.write_text('{"k": {"value": 1, "fingerprint": 5}}')
        store = JSONFileDurableStorage(path)
        assert store.load('k') == StoredEntry(value=1, fingerprint=None)

    def test_json_file_cleanup_is_a_no_op_when_absent(self, tmp_path: Path) -> None:
        JSONFileDurableStorage(tmp_path / 'missing.json').cleanup()

    def test_a_custom_backend_satisfies_the_protocol(self) -> None:
        from pydantic_ai_harness.airflow import DurableStorage

        assert isinstance(InMemoryDurableStorage(), DurableStorage)


class TestFingerprintHelpers:
    def test_model_request_fingerprint_is_stable_and_content_sensitive(self) -> None:
        from pydantic_ai.models import ModelRequestParameters

        request_params = ModelRequestParameters()
        base = fingerprint_model_request('m', [], None, request_params)
        assert base == fingerprint_model_request('m', [], None, request_params)
        assert base != fingerprint_model_request('other', [], None, request_params)

    def test_model_request_fingerprint_is_none_when_unserializable(self) -> None:
        from pydantic_ai.models import ModelRequestParameters

        class Unserializable:
            pass

        # A settings value Pydantic AI would not strip and cannot JSON-encode forces a None fingerprint.
        settings: Any = {'extra_body': Unserializable()}
        assert fingerprint_model_request('m', [], settings, ModelRequestParameters()) is None

    def test_tool_call_fingerprint_includes_the_call_id(self) -> None:
        assert fingerprint_tool_call('act', {}, 'id1') != fingerprint_tool_call('act', {}, 'id2')
        assert fingerprint_tool_call('act', {'a': 1}, 'id1') == fingerprint_tool_call('act', {'a': 1}, 'id1')

    def test_tool_call_fingerprint_is_none_when_unserializable(self) -> None:
        class Unserializable:
            pass

        assert fingerprint_tool_call('act', {'x': Unserializable()}, 'id1') is None

    def test_token_fingerprint_is_stable(self) -> None:
        assert fingerprint_token('calc') == fingerprint_token('calc')
        assert fingerprint_token('calc') != fingerprint_token('other')

    def test_content_settings_strips_transport_only_keys(self) -> None:
        assert _content_settings(None) is None
        assert _content_settings({'timeout': 5}) is None  # transport-only, so nothing left
        assert _content_settings({'temperature': 0.5}) == {'temperature': 0.5}

    def test_strip_message_drops_volatile_keys_and_tolerates_missing_parts(self) -> None:
        # A message-level dict without `parts` still has its volatile ids stripped.
        assert _strip_message({'kind': 'request', 'run_id': 'r', 'timestamp': 't'}) == {'kind': 'request'}
        assert _strip_message({'parts': [{'content': 'hi', 'timestamp': 't'}]}) == {'parts': [{'content': 'hi'}]}


class TestConstructionTimeToolsets:
    def test_non_leaf_construction_toolset_passes_through(self) -> None:
        store = InMemoryDurableStorage()
        external = ExternalToolset[object]([ToolDefinition(name='remote')], id='ext')
        agent = Agent(tool_then_text(), name='a', toolsets=[external], capabilities=[AirflowDurability(storage=store)])

        @agent.tool_plain
        def act() -> str:
            return 'sunny'

        assert agent.run_sync('go').output == 'done'
