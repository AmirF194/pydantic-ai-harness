import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pytest
from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    RequestUsage,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

pytest.importorskip('ddgs')
pytest.importorskip('markdownify')

from inline_snapshot import snapshot
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModelSettings

from pydantic_ai_harness.researcher import Researcher

if TYPE_CHECKING:

    def IsDatetime(*args: Any, **kwargs: Any) -> datetime: ...
    def IsInstance(expected_type: type[RequestUsage], **kwargs: Any) -> RequestUsage: ...
    def IsStr(*args: Any, **kwargs: Any) -> str: ...
else:
    from dirty_equals import IsDatetime, IsInstance, IsStr

pytestmark = pytest.mark.anyio


@pytest.mark.vcr
async def test_researcher_completes_task(allow_model_requests: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OPENAI_API_KEY', os.environ.get('OPENAI_API_KEY', 'replay-key'))
    agent = Agent(
        'openai:gpt-5.6-sol',
        capabilities=[Researcher()],
        model_settings=OpenAIResponsesModelSettings(openai_reasoning_effort='none'),
    )

    result = await agent.run(
        'Compare the free-threaded Python support in CPython 3.13 and 3.14 using official sources. '
        'Delegate the 3.13 investigation to the researcher sub-agent while you investigate 3.14, '
        'then synthesize the maturity status and main limitations. Search for and read the sources '
        'needed for each version, and include direct links.'
    )

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Compare the free-threaded Python support in CPython 3.13 and 3.14 using official sources. Delegate the 3.13 investigation to the researcher sub-agent while you investigate 3.14, then synthesize the maturity status and main limitations. Search for and read the sources needed for each version, and include direct links.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
Search broadly before drawing conclusions.
Read the sources that support each important claim.
Prefer primary and authoritative sources.
Cite every factual claim with a direct source link.
Distinguish sourced facts from your own inference.

You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- researcher: Research a focused sub-question on the web and report back with findings and source links\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='delegate_task',
                        args='{"agent_name":"researcher","task":"Investigate free-threaded Python support specifically in CPython/Python 3.13 using only official primary sources (python.org, docs.python.org, peps.python.org, official CPython GitHub if necessary). Search broadly and read the supporting sources. Report: maturity/experimental status, how users obtain/enable it, ABI/build-tag details, extension compatibility behavior and GIL fallback, known limitations (performance, specialization/JIT, immortalization, memory use, unsupported features), and any relevant packaging/ecosystem implications. Include direct source URLs for every factual claim and distinguish sourced facts from inference. Do not cover 3.14 except where an official source directly compares versions."}',
                        tool_call_id='call_qA5Ef9pl7orsjfymFbkdkQpI',
                        id='fc_0e22809e58aac363006a7f800cb2308190a4f81bda7ad6836d',
                        provider_name='openai',
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='gpt-5.6-sol',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                provider_response_id='resp_0e22809e58aac363006a7f800a80f88190b7ba824115e837df',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='delegate_task',
                        content=IsStr(
                            regex="""\
\\[Tool\\ output\\ too\\ large\\ \\(18,734\\ chars\\);\\ stored\\ to\\ handle\\ '01[a-z0-9]{6}(?:\\-[a-z0-9]{4}){3}\\-[a-z0-9]{12}/call_qA5Ef9pl7orsjfymFbkdkQpI\\.0'\\.\\ Read\\ it\\ with\\ read_tool_result\\(handle='01[a-z0-9]{6}(?:\\-[a-z0-9]{4}){3}\\-[a-z0-9]{12}/call_qA5Ef9pl7orsjfymFbkdkQpI\\.0',\\ offset=0,\\ limit=200,\\ from_end=False,\\ pattern=None\\)\\.\\]\\
\\#\\ CPython\\ 3\\.13\\ free\\-threaded\\ support\\
\\
\\#\\#\\ Executive\\ summary\\
\\
\\*\\*Sourced\\ fact:\\*\\*\\ CPython\\ 3\\.13’s\\ free\\-threaded\\ build\\ is\\ \\*\\*experimental,\\ optional,\\ and\\ not\\ enabled\\ in\\ the\\ normal\\ CPython\\ build\\*\\*\\.\\ PEP\\ 703\\ itself\\ is\\ accepted/final,\\ but\\ the\\ shipped\\ 3\\.13\\ implementation\\ is\\ still\\ explicitly\\ described\\ as\\ experimental,\\ with\\ expected\\ bugs\\ and\\ substantial\\ single\\-threaded\\ overhead\\.\\ \\(\\[peps\\.python\\.org\\]\\(https://peps\\.python\\.org/pep\\-0703/\\)\\)\\ \\ \\
Sources:\\ \\ \\
<https://docs\\.python\\.org/3\\.13/howto/free\\-threading\\-python\\.html>\\ \\ \\
\\.\\.\\.\\[17,734\\ chars\\ omitted\\]\\.\\.\\.\\
ion\\ opt\\-in,\\ and\\ substantive\\ thread\\-safety\\ work\\.\\ Unsupported\\ extensions\\ cause\\ a\\ process\\-wide\\ GIL\\ fallback\\ unless\\ the\\ user\\ forcibly\\ overrides\\ it\\.\\ The\\ most\\ consequential\\ 3\\.13\\ limitations\\ are\\ approximately\\ 40%\\ Python\\-code\\ overhead,\\ disabled\\ specialization\\ and\\ effectively\\ unavailable\\ JIT\\ acceleration,\\ broad\\ immortalization\\ with\\ possible\\ memory\\ growth,\\ unsafe\\ cross\\-thread\\ frame\\ access,\\ unsafe\\ shared\\ iterators,\\ no\\ Stable\\ ABI/Limited\\ API\\ distribution\\ target,\\ and\\ restrictions\\ on\\ custom\\ memory\\ allocators\\.\
"""
                        ),
                        tool_call_id='call_qA5Ef9pl7orsjfymFbkdkQpI',
                        metadata={
                            'overflow_handle': IsStr(
                                regex='01[a-z0-9]{6}(?:\\-[a-z0-9]{4}){3}\\-[a-z0-9]{12}/call_qA5Ef9pl7orsjfymFbkdkQpI\\.0'
                            ),
                            'overflow_bytes': 18769,
                        },
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
Search broadly before drawing conclusions.
Read the sources that support each important claim.
Prefer primary and authoritative sources.
Cite every factual claim with a direct source link.
Distinguish sourced facts from your own inference.

You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- researcher: Research a focused sub-question on the web and report back with findings and source links\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    NativeToolCallPart(
                        tool_name='web_search',
                        args={
                            'type': 'search',
                            'queries': [
                                'site:docs.python.org/3.14 howto free threading Python 3.14',
                                'site:docs.python.org/3.14 whatsnew 3.14 free-threaded Python',
                                'site:peps.python.org free-threaded Python 3.14 PEP 779',
                                'site:python.org Python 3.14 free-threaded officially supported',
                            ],
                            'query': 'site:docs.python.org/3.14 howto free threading Python 3.14',
                        },
                        tool_call_id='ws_0e22809e58aac363006a7f80b8b4fc819098bae51ad0c49d36',
                        id='ws_0e22809e58aac363006a7f80b8b4fc819098bae51ad0c49d36',
                        provider_name='openai',
                    ),
                    NativeToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0e22809e58aac363006a7f80b8b4fc819098bae51ad0c49d36',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    NativeToolCallPart(
                        tool_name='web_search',
                        args={
                            'type': 'search',
                            'queries': [
                                'site:docs.python.org/3.14/howto free-threading-extensions limited API stable ABI free-threaded',
                                'site:docs.python.org/3.14/c-api free-threaded Py_GIL_DISABLED extension module GIL',
                                'site:packaging.python.org free-threaded cp314t wheel tag official',
                                'site:peps.python.org cp314t free threaded ABI tag',
                            ],
                            'query': 'site:docs.python.org/3.14/howto free-threading-extensions limited API stable ABI free-threaded',
                        },
                        tool_call_id='ws_0e22809e58aac363006a7f80bc82cc8190a83f00c9acb010e9',
                        id='ws_0e22809e58aac363006a7f80bc82cc8190a83f00c9acb010e9',
                        provider_name='openai',
                    ),
                    NativeToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0e22809e58aac363006a7f80bc82cc8190a83f00c9acb010e9',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    NativeToolCallPart(
                        tool_name='web_search',
                        args={'type': 'open_page', 'url': 'https://peps.python.org/pep-0779/'},
                        tool_call_id='ws_0e22809e58aac363006a7f80c029208190ade969ee5c3622b7',
                        id='ws_0e22809e58aac363006a7f80c029208190ade969ee5c3622b7',
                        provider_name='openai',
                    ),
                    NativeToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0e22809e58aac363006a7f80c029208190ade969ee5c3622b7',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    NativeToolCallPart(
                        tool_name='web_search',
                        args={
                            'type': 'open_page',
                            'url': 'https://docs.python.org/3.13/howto/free-threading-python.html',
                        },
                        tool_call_id='ws_0e22809e58aac363006a7f80c2c0dc8190a896243925e3b3b2',
                        id='ws_0e22809e58aac363006a7f80c2c0dc8190a896243925e3b3b2',
                        provider_name='openai',
                    ),
                    NativeToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0e22809e58aac363006a7f80c2c0dc8190a896243925e3b3b2',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    ToolCallPart(
                        tool_name='read_tool_result',
                        args='{"handle":"01a0020c-26d3-7125-9757-cd9c6e45489d/call_qA5Ef9pl7orsjfymFbkdkQpI.0","offset":0,"limit":120}',
                        tool_call_id='call_A0K8Cm58j9hDaLmoqUTZxaVI',
                        id='fc_0e22809e58aac363006a7f80c5f7ac8190a53170a0a6f37d5a',
                        provider_name='openai',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='gpt-5.6-sol',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                provider_response_id='resp_0e22809e58aac363006a7f80b784e48190b124434527b43109',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='read_tool_result',
                        content="""\
[handle '01a0020c-26d3-7125-9757-cd9c6e45489d/call_qA5Ef9pl7orsjfymFbkdkQpI.0': 306 matching line(s); showing 120]
# CPython 3.13 free-threaded support

## Executive summary

**Sourced fact:** CPython 3.13’s free-threaded build is **experimental, optional, and not enabled in the normal CPython build**. PEP 703 itself is accepted/final, but the shipped 3.13 implementation is still explicitly described as experimental, with expected bugs and substantial single-threaded overhead. ([peps.python.org](https://peps.python.org/pep-0703/))  \n\
Sources:  \n\
<https://docs.python.org/3.13/howto/free-threading-python.html>  \n\
<https://peps.python.org/pep-0703/>

**Practical assessment — inference:** Treat 3.13 free-threading as a testing, porting, and evaluation target rather than a transparent production replacement for ordinary CPython. That inference follows from its experimental designation, approximately 40% Python-code overhead, disabled specialization, native-extension ABI split, automatic GIL fallback, and documented crash-prone cases.

---

## 1. Maturity and experimental status

- **CPython 3.13 as a release is stable, but its free-threaded build mode is experimental.** It is not enabled by default, may contain bugs, and has a substantial single-threaded performance penalty. ([python.org](https://www.python.org/downloads/release/python-3130/?utm_source=openai))  \n\
  Sources:  \n\
  <https://docs.python.org/3.13/howto/free-threading-python.html>  \n\
  <https://www.python.org/downloads/release/python-3130/>

- **PEP 703 is “Final” and targets Python 3.13.** This means the proposal was accepted and implemented; it does **not** override the 3.13 documentation’s experimental-status warning. The Steering Council’s acceptance also explicitly provided for gradual rollout and possible rollback if disruption proved excessive. ([peps.python.org](https://peps.python.org/pep-0703/))  \n\
  Source:  \n\
  <https://peps.python.org/pep-0703/>

- **The capability is a separate CPython build configuration**, not merely a switch that turns an ordinary 3.13 installation into free-threaded CPython. The free-threaded executable is normally named `python3.13t` or `python3.13t.exe`. ([github.com](https://github.com/python/cpython/blob/main/Doc/whatsnew/3.13.rst?utm_source=openai))  \n\
  Source:  \n\
  <https://docs.python.org/3.13/whatsnew/3.13.html#free-threaded-cpython>

---

## 2. How users obtain and enable it

### Official binaries

- **Windows:** Choose **Customize installation**, then select **Download free-threaded binaries**. This installs additional binaries beside the normal installation. The principal executable is `python3.13t.exe`; the runtime is registered under the `3.13t` tag and can be selected with `py.exe -3.13t`. Command-line installation can use `Include_freethreaded=1`. ([peps.python.org](https://peps.python.org/pep-0703/))  \n\
  Source:  \n\
  <https://docs.python.org/3.13/using/windows.html#installing-free-threaded-binaries>

- **macOS:** Choose **Customize** in the python.org installer and enable the **Free-threaded Python** package. It is not installed by default. It installs a separate `PythonT.framework` and normally provides `/usr/local/bin/python3.13t` and `python3.13t-config`. ([peps.python.org](https://peps.python.org/pep-0703/))  \n\
  Source:  \n\
  <https://docs.python.org/3.13/using/mac.html#installing-free-threaded-binaries>

- **macOS package environments are separate:** the traditional and free-threaded builds have separate search paths, `site-packages` directories, and pip installations. Packages needed in both may therefore have to be installed twice. ([peps.python.org](https://peps.python.org/pep-0703/))  \n\
  Source:  \n\
  <https://docs.python.org/3.13/using/mac.html#installing-free-threaded-binaries>

### Source build

- Build CPython using:

  ```bash
  ./configure --disable-gil
  make
  ```

  `--disable-gil` defines `Py_GIL_DISABLED` and adds `t` to `sys.abiflags`. ([peps.python.org](https://peps.python.org/pep-0703/))  \n\
  Source:  \n\
  <https://docs.python.org/3.13/using/configure.html#cmdoption-disable-gil>

- **The free-threaded implementation requires CPython’s modified mimalloc allocator.** Python 3.13 bundles that allocator, and the release notes identify it as required for free-threaded mode. ([python.org](https://www.python.org/downloads/release/python-3130/?utm_source=openai))  \n\
  Source:  \n\
  <https://www.python.org/downloads/release/python-3130/>

### Runtime control

A free-threaded-capable executable can still run with the GIL enabled:

```bash
python3.13t -X gil=1 app.py
python3.13t -X gil=0 app.py
```

or:

```bash
PYTHON_GIL=1 python3.13t app.py
PYTHON_GIL=0 python3.13t app.py
```

`-X gil` takes precedence over `PYTHON_GIL`; forcing the GIL off is only available in a build configured with `--disable-gil`. ([docs.python.org](https://docs.python.org/3.13/using/cmdline.html))  \n\
Source:  \n\
<https://docs.python.org/3.13/using/cmdline.html#cmdoption-X>

### Detection

```python
import sys
import sysconfig

print(sys.version)
print(sys._is_gil_enabled())                    # Current process state
print(sysconfig.get_config_var("Py_GIL_DISABLED"))  # Build capability
print(sys.abiflags)
```

- `python -VV` and `sys.version` identify an “experimental free-threading build”.
- `sys._is_gil_enabled()` reports the current runtime GIL state.
- `sysconfig.get_config_var("Py_GIL_DISABLED") == 1` is the recommended build-configuration test. ([docs.python.org](https://docs.python.org/3.13/library/sys.html))  \n\
  Sources:  \n\
  <https://docs.python.org/3.13/howto/free-threading-python.html#identifying-free-threaded-python>  \n\
  <https://docs.python.org/3.13/library/sys.html#sys._is_gil_enabled>

---

## 3. ABI and build-tag details

- **The ordinary and free-threaded 3.13 builds have different ABIs.** PEP 703 states that `--disable-gil` is not ABI-compatible with the standard build; extension authors must target both ABIs where both are supported. ([peps.python.org](https://peps.python.org/pep-0703/))  \n\
  Source:  \n\
  <https://peps.python.org/pep-0703/#python-build-modes>

- The free-threaded ABI uses **`t` for “threading”**:
  - `sys.abiflags` contains `t`;
  - executables are normally `python3.13t`;
  - extension shared libraries, binaries, and wheels use a `t` suffix;
  - POSIX `pkg-config` metadata is named, for example, `python-3.13t.pc`. ([peps.python.org](https://peps.python.org/pep-0703/))  \n\
  Sources:  \n\
  <https://docs.python.org/3.13/using/configure.html#cmdoption-disable-gil>  \n\
  <https://docs.python.org/3.13/howto/free-threading-extensions.html#building-extensions-for-the-free-threaded-build>  \n\
  <https://docs.python.org/3.13/whatsnew/3.13.html#build-changes>
""",
                        tool_call_id='call_A0K8Cm58j9hDaLmoqUTZxaVI',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
Search broadly before drawing conclusions.
Read the sources that support each important claim.
Prefer primary and authoritative sources.
Cite every factual claim with a direct source link.
Distinguish sourced facts from your own inference.

You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- researcher: Research a focused sub-question on the web and report back with findings and source links\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='read_tool_result',
                        args='{"handle":"01a0020c-26d3-7125-9757-cd9c6e45489d/call_qA5Ef9pl7orsjfymFbkdkQpI.0","offset":120,"limit":160}',
                        tool_call_id='call_w0NBoiNPXQPvdSdmFB5hOJjO',
                        id='fc_0e22809e58aac363006a7f80c8a71481909eaf7c3b8b0f4e7d',
                        provider_name='openai',
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='gpt-5.6-sol',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                provider_response_id='resp_0e22809e58aac363006a7f80c74fdc8190a6f32b82db00d6d5',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='read_tool_result',
                        content="""\
[handle '01a0020c-26d3-7125-9757-cd9c6e45489d/call_qA5Ef9pl7orsjfymFbkdkQpI.0': 306 matching line(s); showing 160]
- **Native extensions must be compiled specifically for free-threaded CPython.** A binary built for normal CPython 3.13 is not the free-threaded extension ABI merely because the free-threaded interpreter can restore the GIL. ([peps.python.org](https://peps.python.org/pep-0703/))  \n\
  Sources:  \n\
  <https://docs.python.org/3.13/howto/free-threading-extensions.html#building-extensions-for-the-free-threaded-build>  \n\
  <https://peps.python.org/pep-0703/#backwards-compatibility>

- **The 3.13 free-threaded build does not support the Limited C API or Stable ABI for extension distribution.** Separate free-threaded wheels are required; an existing `abi3` wheel strategy continues to cover non-free-threaded versions only.   \n\
  Source:  \n\
  <https://docs.python.org/3.13/howto/free-threading-extensions.html#limited-c-api-and-stable-abi>

- `Py_GIL_DISABLED` is defined as `1` in free-threaded builds and is absent in normal builds. On Windows, the 3.13 extension HOWTO warns that it may need to be manually supplied when compiling extensions from source because of an installer limitation.   \n\
  Source:  \n\
  <https://docs.python.org/3.13/howto/free-threading-extensions.html#identifying-the-free-threaded-build-in-c>

**Inference:** In wheel terminology, the `t` ABI distinction is commonly exposed as a `cp313t`-style compatibility tag. The authoritative 3.13 claims are the documented `t` ABI suffix and requirement for separate wheels; tooling should derive the exact supported tags from the interpreter rather than hard-code them.

---

## 4. Extension compatibility and automatic GIL fallback

### Declaring compatibility

A C extension must explicitly declare that it does not need the GIL.

For multi-phase initialization:

```c
static PyModuleDef_Slot slots[] = {
#if PY_VERSION_HEX >= 0x030D0000
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL}
};
```

For single-phase initialization, call:

```c
#ifdef Py_GIL_DISABLED
PyUnstable_Module_SetGIL(module, Py_MOD_GIL_NOT_USED);
#endif
```

`Py_mod_gil` defaults to `Py_MOD_GIL_USED` if omitted. `PyUnstable_Module_SetGIL()` is part of the Unstable API and is only available in free-threaded builds. ([docs.python.org](https://docs.python.org/3.13/c-api/module.html))  \n\
Sources:  \n\
<https://docs.python.org/3.13/howto/free-threading-extensions.html#module-initialization>  \n\
<https://docs.python.org/3.13/c-api/module.html#c.Py_mod_gil>  \n\
<https://docs.python.org/3.13/c-api/module.html#c.PyUnstable_Module_SetGIL>

### Fallback behavior

- If an imported C extension does not explicitly declare free-threaded safety, CPython emits a warning and **enables the GIL for the process**. PEP 703 describes this as pausing the threads and enabling the GIL before continuing. ([peps.python.org](https://peps.python.org/pep-0703/))  \n\
  Sources:  \n\
  <https://docs.python.org/3.13/howto/free-threading-python.html#the-global-interpreter-lock-in-free-threaded-python>  \n\
  <https://peps.python.org/pep-0703/#py-mod-gil-slot>

- Users can override this fallback with `PYTHON_GIL=0` or `-X gil=0`. This does **not** make an unsafe extension thread-safe; it only instructs CPython not to restore the GIL. ([docs.python.org](https://docs.python.org/3.13/using/cmdline.html))  \n\
  Sources:  \n\
  <https://docs.python.org/3.13/using/cmdline.html#envvar-PYTHON_GIL>  \n\
  <https://docs.python.org/3.13/whatsnew/3.13.html#free-threaded-cpython>

### Source-level extension changes

Merely setting `Py_MOD_GIL_NOT_USED` is not sufficient unless the extension is genuinely thread-safe:

- State formerly protected implicitly by the GIL may need explicit locks or thread-local storage.
- Direct access to concurrently mutable C-API struct fields is unsafe.
- Fast macros such as `PyList_GET_ITEM` do not lock.
- Borrowed-reference APIs may be unsafe if another thread can mutate the owning container; new strong-reference APIs such as `PyList_GetItemRef()` and `PyDict_GetItemRef()` are provided.
- `PyDict_Next()` does not lock and may need a critical section.
- The object/memory allocation-domain distinction is a hard requirement: Python objects must use the object domain, while non-object buffers should not use `PyObject_Malloc()`.   \n\
  Source:  \n\
  <https://docs.python.org/3.13/howto/free-threading-extensions.html>

- Thread-state/GIL APIs such as `PyGILState_Ensure()`, `PyEval_SaveThread()`, and `Py_BEGIN_ALLOW_THREADS` are still relevant because they also manage attached thread state and allow the cyclic collector to run around blocking operations.   \n\
  Source:  \n\
  <https://docs.python.org/3.13/howto/free-threading-extensions.html#thread-state-and-gil-apis>

---

## 5. Known limitations

### Performance and specialization

- **The official 3.13 HOWTO reports approximately 40% overhead on `pyperformance` for the free-threaded build versus the ordinary GIL-enabled build.** Workloads dominated by C extensions or I/O may experience less impact.   \n\
  Source:  \n\
  <https://docs.python.org/3.13/howto/free-threading-python.html#single-threaded-performance>

- The largest cited cause is that **PEP 659’s specializing adaptive interpreter is disabled in the 3.13 free-threaded build**.   \n\
  Source:  \n\
  <https://docs.python.org/3.13/howto/free-threading-python.html#single-threaded-performance>

- PEP 703 contains older reference-implementation estimates of roughly 5–8% overhead, but those are proposal-era measurements and should not be substituted for the shipped 3.13 documentation’s approximately 40% figure. ([peps.python.org](https://peps.python.org/pep-0703/))  \n\
  Source:  \n\
  <https://peps.python.org/pep-0703/#performance>

### JIT

- **Sourced facts:** The 3.13 JIT/Tier-2 pipeline starts from specialized Tier-1 bytecode, while specialization is disabled in 3.13 free-threaded mode. A CPython maintainer also confirmed in the official issue tracker that when the GIL is disabled, the JIT is disabled at runtime in 3.13. ([github.com](https://github.com/python/cpython/blob/main/Doc/whatsnew/3.13.rst?utm_source=openai))  \n\
  Sources:  \n\
  <https://docs.python.org/3.13/whatsnew/3.13.html#an-experimental-just-in-time-jit-compiler>  \n\
  <https://github.com/python/cpython/issues/129438>

- **Inference:** Building 3.13 with both `--disable-gil` and `--enable-experimental-jit` does not provide a useful free-threaded JIT configuration. The source build may accept both options, but free-threaded execution does not obtain the JIT benefit.

### Immortalization and memory use

After the first additional thread starts, 3.13 immortalizes:

- module-level functions;
- method descriptors;
- code objects;
- module objects and module dictionaries;
- classes/type objects.

Numeric and string literals and strings returned by `sys.intern()` are also immortalized. These objects are never deallocated and their reference counts are not modified. Programs that dynamically create many such objects may therefore retain significantly more memory.   \n\
Source:  \n\
<https://docs.python.org/3.13/howto/free-threading-python.html#immortalization>

**Inference:** Dynamic module/plugin loaders, notebook-like systems, code-generation systems, and applications that repeatedly construct classes are more exposed to this retention than applications with a mostly fixed module and class graph.

### Frame objects

Accessing another thread’s frame objects is unsafe and may crash the process. Consequently, `sys._current_frames()` is generally unsafe in 3.13 free-threaded mode. `inspect.currentframe()` and `sys._getframe()` are generally safe only while the resulting frame remains within its originating thread.   \n\
Source:  \n\
<https://docs.python.org/3.13/howto/free-threading-python.html#frame-objects>

### Iterators

Sharing a single iterator object between threads is generally unsafe. The official documentation warns of duplicate or missing elements and possible interpreter crashes.   \n\
Source:  \n\
<https://docs.python.org/3.13/howto/free-threading-python.html#iterators>

### Built-in container locking is not a language guarantee

`dict`, `list`, and `set` use internal locking to approximate the safety behavior provided by the GIL, but Python has not historically guaranteed particular behavior for concurrent modification. Applications should use `threading.Lock` or other explicit synchronization rather than treating the current internal locking as a durable API contract.   \n\
Source:  \n\
<https://docs.python.org/3.13/howto/free-threading-python.html#thread-safety>

### Memory allocators

PEP 703 says wholesale replacement of CPython’s allocator with an allocator such as jemalloc or tcmalloc is not compatible with the free-threaded design; custom allocators must delegate actual allocation to the previously installed allocator. ([peps.python.org](https://peps.python.org/pep-0703/))  \n\
Source:  \n\
<https://peps.python.org/pep-0703/#backwards-compatibility>

### Object finalization timing

PEP 703 documents that destructors and weak-reference callbacks for some code objects and top-level functions may be delayed until cyclic collection, and destruction of some cross-thread objects may be delayed slightly by biased reference counting. ([peps.python.org](https://peps.python.org/pep-0703/))  \n\
Source:  \n\
<https://peps.python.org/pep-0703/#backwards-compatibility>

---

## 6. Packaging and ecosystem implications

- **pip 24.1 or newer is required to install packages containing C extensions in a free-threaded 3.13 build.** ([github.com](https://github.com/python/cpython/blob/main/Doc/whatsnew/3.13.rst?utm_source=openai))  \n\
  Source:  \n\
  <https://docs.python.org/3.13/whatsnew/3.13.html#free-threaded-cpython>

- **Native projects need another wheel/build dimension:** operating system × architecture × Python feature version × ordinary/free-threaded ABI. A normal 3.13 wheel or `abi3` wheel does not satisfy the 3.13 free-threaded ABI requirement. ([peps.python.org](https://peps.python.org/pep-0703/))  \n\
  Sources:  \
""",
                        tool_call_id='call_w0NBoiNPXQPvdSdmFB5hOJjO',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
Search broadly before drawing conclusions.
Read the sources that support each important claim.
Prefer primary and authoritative sources.
Cite every factual claim with a direct source link.
Distinguish sourced facts from your own inference.

You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- researcher: Research a focused sub-question on the web and report back with findings and source links\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    NativeToolCallPart(
                        tool_name='web_search',
                        args={'type': 'open_page'},
                        tool_call_id='ws_0e22809e58aac363006a7f80cb11d08190949f250000e75371',
                        id='ws_0e22809e58aac363006a7f80cb11d08190949f250000e75371',
                        provider_name='openai',
                    ),
                    NativeToolReturnPart(
                        tool_name='web_search',
                        content={'status': 'completed'},
                        tool_call_id='ws_0e22809e58aac363006a7f80cb11d08190949f250000e75371',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content="""\
## Bottom line

**CPython 3.13 introduced free-threading as an experimental preview; CPython 3.14 promotes it to an officially supported—but still optional—build configuration.** The 3.14 implementation is substantially more mature and much faster for single-threaded code, but it is **not yet the default CPython build**, does not eliminate the need for application-level synchronization, and still has meaningful extension-packaging and memory-use limitations. ([PEP 779](https://peps.python.org/pep-0779/))

| Area | CPython 3.13 | CPython 3.14 |
|---|---|---|
| Maturity | Experimental, phase I | Officially supported, phase II |
| Default build | No | No |
| Single-thread overhead | About 40% on `pyperformance` | About 1–8% in current 3.14 documentation; the original 3.14 release notes described roughly 5–10% |
| Adaptive specialization | Disabled | Enabled |
| Implementation status | Initial implementation with temporary workarounds | PEP 703 implementation described as finished; temporary workarounds replaced |
| Broad object immortalization | Yes | Mostly removed; limited to code constants and explicitly interned strings |
| Native extensions | Separate free-threaded build; incompatible extensions restore the GIL | Same fundamental model |
| Stable/Limited ABI | Not supported for free-threaded extensions | Still not supported for free-threaded extensions |
| Frame/iterator hazards | Broad and potentially crash-prone | Improved frame handling, but important hazards remain |
| Memory use | Potentially severe retention from broad immortalization | Still typically higher than regular CPython, for several architectural reasons |
| Free-threaded JIT | Effectively unavailable | JIT compilation remains unsupported |

## 1. Maturity

### Python 3.13: experimental rollout

The 3.13 free-threaded build is explicitly experimental. It is a distinct CPython build configuration, obtained through optional Windows/macOS installer components or by configuring a source build with `--disable-gil`. It is normally identified by a `t` ABI suffix and executable names such as `python3.13t`. ([3.13 free-threading HOWTO](https://docs.python.org/3.13/howto/free-threading-python.html), [PEP 703](https://peps.python.org/pep-0703/))

**Practical interpretation:** 3.13 is best treated as a porting, testing, and evaluation target. Its experimental label, high serial overhead, disabled specialization, extension fallback behavior, and documented crash cases make it a poor transparent substitute for ordinary CPython.

### Python 3.14: supported but optional

PEP 779 moves free-threaded CPython into **phase II**: it is officially supported and no longer experimental, with the API design regarded as stable enough to follow normal compatibility policy. Phase III—making it the default or only build—remains undecided. ([PEP 779](https://peps.python.org/pep-0779/), [What’s New in Python 3.14](https://docs.python.org/3.14/whatsnew/3.14.html))

“Supported” therefore means that users and vendors can rely on the configuration continuing to exist and receiving normal maintenance. It does **not** mean that all Python packages support it, that it is selected automatically, or that the ecosystem has converged on it as the normal CPython runtime.

## 2. Performance and implementation maturity

### 3.13

The official 3.13 HOWTO reports approximately **40% single-threaded overhead** on `pyperformance`. The largest stated cause is that the PEP 659 specializing adaptive interpreter is disabled in the free-threaded build. Programs dominated by I/O or native-extension execution may experience less overhead. ([3.13 known limitations](https://docs.python.org/3.13/howto/free-threading-python.html#single-threaded-performance))

### 3.14

In 3.14, the PEP 703 implementation is described as finished, including its C API work; temporary interpreter workarounds were replaced with more permanent mechanisms, and the specializing adaptive interpreter is enabled. The 3.14 release notes reported roughly **5–10%** serial overhead, while the current 3.14 HOWTO reports benchmark averages ranging from about **1% on macOS AArch64 to 8% on x86-64 Linux**. These are workload- and platform-dependent measurements rather than universal guarantees. ([3.14 free-threading improvements](https://docs.python.org/3.14/whatsnew/3.14.html#free-threaded-mode-improvements), [3.14 known limitations](https://docs.python.org/3.14/howto/free-threading-python.html#single-threaded-performance))

PEP 779 also recorded approximately **15–20% greater memory use** as a geometric mean in its evaluation and treated some extra memory consumption as an inherent tradeoff for efficient free-threading. ([PEP 779 rationale](https://peps.python.org/pep-0779/#rationale))

**Synthesis:** The reduction from roughly 40% to single-digit serial overhead is the largest practical maturity improvement in 3.14. It changes free-threading from primarily an experiment into a credible deployment option where parallel Python execution provides enough benefit.

## 3. Extensions, GIL fallback, and packaging

Both releases use essentially the same compatibility model:

- A free-threaded interpreter may be run with the GIL enabled using `PYTHON_GIL` or `-X gil`.
- A C extension must explicitly declare that it supports operation without the GIL.
- Importing an extension that does not make that declaration causes CPython to issue a warning and enable the GIL for the process.
- Forcing `-X gil=0` does not magically make an unsafe extension thread-safe. ([3.13 HOWTO](https://docs.python.org/3.13/howto/free-threading-python.html#the-global-interpreter-lock-in-free-threaded-python), [3.14 HOWTO](https://docs.python.org/3.14/howto/free-threading-python.html#the-global-interpreter-lock-in-free-threaded-python))

Native extensions must be compiled specifically for the free-threaded ABI. Their executables, shared libraries, and wheels use the `t` distinction—commonly represented by tags such as `cp313t` and `cp314t`. ([3.14 extension HOWTO](https://docs.python.org/3.14/howto/free-threading-extensions.html#building-extensions-for-the-free-threaded-build))

Neither 3.13 nor 3.14 supports the Limited C API or Stable ABI for free-threaded extensions. Projects consequently need version-specific free-threaded wheels; an ordinary `abi3` wheel does not cover these runtimes. ([3.13 extension HOWTO](https://docs.python.org/3.13/howto/free-threading-extensions.html#limited-c-api-and-stable-abi), [3.14 extension HOWTO](https://docs.python.org/3.14/howto/free-threading-extensions.html#limited-c-api-and-stable-abi))

On Windows in 3.14, an extension build backend must explicitly define `Py_GIL_DISABLED`; the C compiler no longer infers it automatically. ([3.14 release notes](https://docs.python.org/3.14/whatsnew/3.14.html#free-threaded-mode-improvements))

**Practical limitation:** A single incompatible native dependency can restore the GIL process-wide, eliminating the main benefit of the free-threaded runtime. Package availability must therefore be checked across the application’s entire native dependency graph.

## 4. Thread safety is not automatic

In both releases, built-ins such as `dict`, `list`, and `set` use internal locks to provide behavior broadly similar to GIL-enabled CPython. However, that behavior is documented as an implementation description, not a language-level guarantee for concurrent mutation. Explicit synchronization such as `threading.Lock` remains recommended. ([3.14 thread-safety guidance](https://docs.python.org/3.14/howto/free-threading-python.html#thread-safety))

Extension authors face additional work:

- GIL-protected global or module state needs explicit synchronization.
- Borrowed-reference APIs can become unsafe if another thread mutates the owner.
- Direct field access and “fast” macros may bypass locking.
- Critical sections or ordinary native mutexes may be required.
- A thread must still have an attached Python thread state before accessing Python objects, even though attaching one no longer normally means acquiring the GIL. ([3.14 extension HOWTO](https://docs.python.org/3.14/howto/free-threading-extensions.html), [3.14 thread-state documentation](https://docs.python.org/3.14/c-api/threads.html))

**Inference:** Free-threading removes the GIL as a serialization mechanism; it does not turn existing race-prone code into correctly synchronized parallel code.

## 5. Memory and immortalization

### 3.13

When a second thread starts, 3.13 immortalizes several broad categories, including module-level functions, method descriptors, code objects, modules and module dictionaries, and classes. Numeric and string literals and strings returned by `sys.intern()` are also immortal. Because these objects are never deallocated, systems that repeatedly generate classes, modules, or functions can accumulate memory. ([3.13 immortalization limitation](https://docs.python.org/3.13/howto/free-threading-python.html#immortalization))

### 3.14

The broad 3.13 scheme was substantially reduced. In 3.14, immortalization is limited to:

- Code constants, including numeric, string, and qualifying tuple literals.
- Strings interned using `sys.intern()`.

([3.14 immortalization limitation](https://docs.python.org/3.14/howto/free-threading-python.html#immortalization))

Memory use can nevertheless remain higher because:

- Non-GC objects have larger headers.
- Free-threaded CPython uses mimalloc rather than pymalloc.
- QSBR can defer reclamation.
- Biased, deferred, and per-thread reference-counting mechanisms can cause objects to be reclaimed later.
- Interned strings remain alive until interpreter shutdown.

([3.14 memory-use discussion](https://docs.python.org/3.14/howto/free-threading-python.html#increased-memory-usage))

## 6. Remaining correctness and feature limitations

### Frames

In 3.13, accessing frame objects belonging to another thread is generally unsafe and may crash, making `sys._current_frames()` generally unsafe in that configuration. ([3.13 frame limitation](https://docs.python.org/3.13/howto/free-threading-python.html#frame-objects))

The 3.14 warning is narrower but still serious: accessing `frame.f_locals` while that frame is executing in another thread can crash the interpreter. ([3.14 frame limitation](https://docs.python.org/3.14/howto/free-threading-python.html#frame-objects))

### Iterators

Sharing an iterator concurrently remains generally unsafe. Threads may observe duplicate or missing elements; 3.13 additionally documents possible interpreter crashes. ([3.13 iterator limitation](https://docs.python.org/3.13/howto/free-threading-python.html#iterators), [3.14 iterator limitation](https://docs.python.org/3.14/howto/free-threading-python.html#iterators))

### JIT

The 3.14 experimental JIT does not support free-threaded builds. Thus, enabling adaptive specialization in 3.14 fixes the major 3.13 performance deficit, but free-threaded users still cannot combine that configuration with JIT compilation. ([3.14 JIT release notes](https://docs.python.org/3.14/whatsnew/3.14.html#binary-releases-for-the-experimental-just-in-time-compiler))

## Overall assessment

- **3.13:** an experimental compatibility and porting platform. It proves that parallel Python execution works, but its roughly 40% serial overhead, broad immortalization, crash-prone introspection cases, and early ecosystem state substantially constrain production adoption.
- **3.14:** the first version that can reasonably be treated as a supported deployment target. Its design and APIs are considered stable, specialization is restored, serial overhead is usually in the single digits, and broad immortalization is removed.
- **Still unresolved in 3.14:** it remains an optional ABI, native extensions need separate builds, incompatible extensions can restore the GIL, there is no free-threaded Stable ABI, memory use is higher, shared mutable state still needs locks, some cross-thread frame and iterator operations remain unsafe, and the JIT is unavailable.

The clearest summary is: **3.13 demonstrates the model; 3.14 operationalizes it—but does not yet make it universal or transparent.**\
""",
                        id='msg_0e22809e58aac363006a7f80ccd1bc819089f77b255f60b6e0',
                        provider_name='openai',
                        provider_details={'phase': 'final_answer'},
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='gpt-5.6-sol',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                provider_response_id='resp_0e22809e58aac363006a7f80c9a06881909ff638ddf7667856',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert isinstance(result.output, str)
    assert len(result.output) > 200
    tool_names = {
        part.tool_name
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolCallPart | NativeToolCallPart)
    }
    assert 'delegate_task' in tool_names
    assert tool_names & {'web_search', 'web_search_preview'}
    assert tool_names & {'web_fetch', 'open_url', 'read_tool_result'}
