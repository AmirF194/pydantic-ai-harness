"""Integration tests that require a real, running Modal container.

The fake-backed suites already cover the harness-owned logic: timeout quantization,
the output ring buffer, path resolution math, and exception mapping. This live tier
admits only regressions a correctly written fake could not catch: real process
execution, infra-enforced deadlines, one filesystem shared by Modal's file API and
the shell, create-time environment and workdir propagation, and real lifecycle
state in Modal's control plane.

Admission rule:
  A test belongs here only when its docstring can name the fake-encoded assumption
  it validates against real Modal behavior.

Transition map:
  * TestRealExecution: `exec(argv, timeout=...)` becomes `run(..., shell=...)`; `returncode` becomes
    `exit_code`; timeout checks catch `CommandTimeoutError` and assert `.stdout`.
  * TestCreateConfiguration: create-time `env` and `workdir` assertions retarget to the future sandbox
    configuration surface, with probes run through `Sandbox.run(...)`.
  * TestRealFilesystem: byte file operations retarget to `pydantic-sandboxes` file APIs; the relative
    path case becomes explicit `resolve()` / `working_dir()` coverage.
  * TestRealLifecycle: dead sandbox checks retarget from `ModalSandboxUnavailableError` to
    `SandboxNotFoundError`, and attach/reuse maps to the provider-agnostic attach surface.

Gating:
  * `modal_live` marker: CI can select this tier with `-m modal_live`; the normal suite can deselect it.
  * skipped unless `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` are set, or `~/.modal.toml` exists.
  * a module-scoped `anyio_backend` fixture keeps the shared Modal handle on one asyncio loop.

Run locally: `uv run pytest -m modal_live tests/experimental/modal_sandbox/test_modal_live.py`
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import pytest

from pydantic_ai_harness.experimental.modal_sandbox import (
    ModalSandboxError,
    ModalSandboxSession,
    ModalSandboxTerminalError,
    ModalSandboxUnavailableError,
)


def _has_modal_credentials() -> bool:
    has_env_token = os.getenv('MODAL_TOKEN_ID') is not None and os.getenv('MODAL_TOKEN_SECRET') is not None
    return has_env_token or Path('~/.modal.toml').expanduser().exists()


pytestmark = [
    pytest.mark.modal_live,
    pytest.mark.skipif(
        not _has_modal_credentials(),
        reason='requires MODAL_TOKEN_ID / MODAL_TOKEN_SECRET or ~/.modal.toml for a live Modal run',
    ),
]

# A small, common image keeps cold starts cheap; these tests need only a POSIX shell and coreutils.
_IMAGE = 'python:3.12-slim'


def _unique(prefix: str) -> str:
    """Return a collision-resistant path or name segment for a shared live sandbox."""
    return f'{prefix}-{uuid.uuid4().hex}'


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture(scope='module')
async def session() -> AsyncIterator[ModalSandboxSession]:
    """One live owned sandbox shared by exec and filesystem tests.

    Each test writes under `_unique(...)` paths, so the shared container avoids repeated cold starts
    without coupling test state. Lifecycle tests create their own sandboxes because ownership,
    expiry, attach, and termination are the behavior under test there.
    """
    async with ModalSandboxSession(image=_IMAGE, sandbox_timeout=600) as live:
        yield live


class TestRealExecution:
    """Behaviors that only exist because a real process runs on real Modal infra."""

    async def test_runs_a_real_process(self, session: ModalSandboxSession) -> None:
        """Validates the fake-encoded assumption that exec stdout, stderr, and exit code match a process."""
        result = await session.exec(['sh', '-c', 'echo out; echo err 1>&2; exit 3'], timeout=30)

        assert result.stdout.strip() == 'out'
        assert result.stderr.strip() == 'err'
        assert result.returncode == 3
        assert result.timed_out is False

    async def test_timeout_preserves_pre_deadline_output(self, session: ModalSandboxSession) -> None:
        """Validates the fake-encoded assumption that output printed before an infra timeout is preserved."""
        result = await session.exec(['sh', '-c', 'echo DIAGNOSTIC; sleep 30'], timeout=2)

        # Durable assertion: future `pydantic-sandboxes` exposes this on `CommandTimeoutError.stdout`.
        assert 'DIAGNOSTIC' in result.stdout
        assert result.timed_out is True

    async def test_large_stderr_does_not_block_stdout(self, session: ModalSandboxSession) -> None:
        """Validates the fake-encoded assumption that Modal buffers streams without stderr deadlock."""
        result = await session.exec(['sh', '-c', 'seq 1 300000 1>&2; echo done'], timeout=60)

        assert result.returncode == 0
        assert result.stdout == 'done\n'
        stderr_lines = result.stderr.splitlines()
        assert stderr_lines[0] == '1'
        assert stderr_lines[-1] == '300000'
        assert len(stderr_lines) == 300000

    async def test_concurrent_commands_share_one_container(self, session: ModalSandboxSession) -> None:
        """Validates the fake-encoded assumption that one Modal sandbox can multiplex concurrent execs."""
        results: dict[int, str] = {}

        async def run(n: int) -> None:
            out = await session.exec(['sh', '-c', f'echo job-{n}'], timeout=15)
            results[n] = out.stdout.strip()

        async with anyio.create_task_group() as tg:
            for n in range(8):
                tg.start_soon(run, n)

        assert results == {n: f'job-{n}' for n in range(8)}

    async def test_signal_exit_is_not_timeout(self, session: ModalSandboxSession) -> None:
        """Validates the fake-encoded assumption that signal death is a real exit, not Modal's timeout sentinel."""
        result = await session.exec(['sh', '-c', 'kill -KILL $$'], timeout=15)

        assert result.returncode > 128
        assert result.returncode != -1
        assert result.timed_out is False

    async def test_nonexistent_binary_is_wrapped_or_exit_127(self, session: ModalSandboxSession) -> None:
        """Validates the fake-encoded assumption that a missing binary does not leak a raw provider error.

        Characterization canary: this currently accepts either a wrapped `ModalSandboxError` or
        `exit_code == 127`; pin the observed branch after the first real live run confirms Modal's behavior.
        """
        binary = _unique('definitely-not-a-real-binary')

        try:
            result = await session.exec([binary], timeout=15)
        except ModalSandboxError:
            return

        assert result.returncode == 127


class TestCreateConfiguration:
    """Create-time configuration reaching the real process, not only Modal create kwargs."""

    async def test_env_and_workdir_reach_processes(self) -> None:
        """Validates the fake-encoded assumption that create-time `env` and `workdir` reach commands."""
        probe = _unique('live-value')
        async with ModalSandboxSession(
            image=_IMAGE,
            sandbox_timeout=120,
            workdir='/tmp',
            env={'HARNESS_ENV_PROBE': probe},
        ) as session:
            env_result = await session.exec(['sh', '-c', 'printf %s "$HARNESS_ENV_PROBE"'], timeout=15)
            pwd_result = await session.exec(['pwd'], timeout=15)

        assert env_result.stdout == probe
        assert pwd_result.stdout.strip() == '/tmp'


class TestRealFilesystem:
    """One real filesystem shared by Modal's file API and the shell."""

    async def test_shell_and_file_api_see_the_same_filesystem(self, session: ModalSandboxSession) -> None:
        """Validates the fake-encoded assumption that exec and filesystem APIs share one backing fs."""
        api_path = f'/tmp/{_unique("api")}.txt'
        await session.write_bytes(api_path, b'from-file-api\n')
        via_shell = await session.exec(['cat', api_path], timeout=15)
        assert via_shell.stdout == 'from-file-api\n'

        shell_path = f'/tmp/{_unique("shell")}.txt'
        wrote = await session.exec(['sh', '-c', f'printf from-shell > {shell_path}'], timeout=15)
        assert wrote.returncode == 0
        assert await session.read_bytes(shell_path) == b'from-shell'

    async def test_binary_roundtrip_creating_parent_dirs(self, session: ModalSandboxSession) -> None:
        """Validates the fake-encoded assumption that Modal stores raw bytes and creates real parent dirs."""
        path = f'/tmp/{_unique("io")}/nested/deep/data.bin'
        payload = b'\x00\x01hello \xf0\x9f\x9a\x80 world'

        await session.write_bytes(path, payload)

        assert await session.read_bytes(path) == payload

    async def test_large_filesystem_transfer_near_read_limit(self, session: ModalSandboxSession) -> None:
        """Validates the fake-encoded assumption that Modal's beta fs API handles a near-limit transfer."""
        path = f'/tmp/{_unique("big")}.bin'
        payload = b'A' * (4 * 1024 * 1024)

        await session.write_bytes(path, payload)

        assert await session.file_size(path) == len(payload)
        assert await session.read_bytes(path) == payload

    async def test_missing_file_raises_recoverable_error(self, session: ModalSandboxSession) -> None:
        """Validates the fake-encoded assumption that a missing file is recoverable, not a dead sandbox."""
        with pytest.raises(ModalSandboxError) as exc_info:
            await session.read_bytes(f'/tmp/{_unique("missing")}')

        assert not isinstance(exc_info.value, ModalSandboxTerminalError)

    async def test_workdir_and_relative_file_resolution_share_one_view(self) -> None:
        """Validates the fake-encoded assumption that relative file API paths share the process cwd view.

        Future `pydantic-sandboxes` retarget: callers resolve explicitly via `sandbox.resolve()` /
        `working_dir()`, because the provider-agnostic seam rejects implicit relative paths.
        """
        filename = f'{_unique("rel")}.txt'
        async with ModalSandboxSession(image=_IMAGE, sandbox_timeout=120, workdir='/tmp') as session:
            await session.write_bytes(filename, b'from-relative-path\n')
            result = await session.exec(['cat', filename], timeout=15)

        assert result.stdout == 'from-relative-path\n'


class TestRealLifecycle:
    """Provisioning, expiry, teardown, and attach semantics in Modal's real control plane."""

    async def test_sandbox_timeout_expiry_mid_session_is_unavailable(self) -> None:
        """Validates the fake-encoded assumption that real Modal expiry raises an unavailable sandbox error.

        Migration note: the post-expiry command must retarget to `SandboxNotFoundError` (fatal), not base
        `SandboxError` (retryable); this test is the forcing function for that mapping decision in the
        pydantic-sandboxes Modal adapter.
        """
        async with ModalSandboxSession(image=_IMAGE, sandbox_timeout=15) as session:
            await anyio.sleep(25)
            with pytest.raises(ModalSandboxUnavailableError):
                await session.exec(['echo', 'after-expiry'], timeout=15)

    async def test_terminate_actually_destroys_the_container(self) -> None:
        """Validates the fake-encoded assumption that exiting an owned session destroys the real sandbox."""
        async with ModalSandboxSession(image=_IMAGE, sandbox_timeout=120) as owner:
            sandbox_id = owner.sandbox_id
            assert sandbox_id is not None

        became_unavailable = False
        for attempt in range(6):
            try:
                async with ModalSandboxSession(sandbox_id=sandbox_id):
                    pass  # pragma: no cover
            except ModalSandboxUnavailableError:
                became_unavailable = True
                break
            # Modal can lag about 30s before reporting external termination to a fresh attach.
            if attempt < 5:
                await anyio.sleep(5)

        assert became_unavailable

    async def test_attach_reuses_state_and_leaves_container_running(self) -> None:
        """Validates the fake-encoded assumption that attach reuses state and does not terminate ownership."""
        marker = f'/tmp/{_unique("persist")}.txt'
        async with ModalSandboxSession(image=_IMAGE, sandbox_timeout=120) as owner:
            await owner.write_bytes(marker, b'shared')
            sandbox_id = owner.sandbox_id
            assert sandbox_id is not None

            async with ModalSandboxSession(sandbox_id=sandbox_id) as attached:
                assert attached.sandbox_id == sandbox_id
                assert await attached.read_bytes(marker) == b'shared'

            assert (await owner.exec(['cat', marker], timeout=15)).stdout == 'shared'
