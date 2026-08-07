---
runtimes:
  uv: {}
pre-agent-steps:
  - name: Install Pydantic AI Harness CLI
    run: |
      uv tool install --quiet "pydantic-ai-harness==0.1.0"
      pydantic-harness --version
engine:
  id: pydantic-ai
  display-name: Pydantic AI
  description: Pydantic AI Harness coding agent CLI running in non-interactive mode
  experimental: true
  mcp: false
  provider:
    name: github
  behaviors:
    secret-strategy: universal-llm-consumer
    capabilities:
      tools-allowlist: false
      max-turns: true
      web-search: false
    network:
      defaults:
        - host.docker.internal
        - github.com
        - raw.githubusercontent.com
        - api.github.com
        - objects.githubusercontent.com
        - pypi.org
        - files.pythonhosted.org
      provider-domains:
        copilot: api.githubcopilot.com
        anthropic: api.anthropic.com
        openai: api.openai.com
    execution:
      command-name: pydantic-harness
      args:
        - --no-color
      step-name: Execute Pydantic AI Harness CLI
      model-env-var: PYDANTIC_AI_MODEL
      provider-env-mode: universal-llm-consumer
      write-timestamp: true
---

<!--
# Pydantic AI Harness CLI

Shared engine definition for the Pydantic AI Harness coding agent.
Import this file and set `engine: id: pydantic-ai` to use it.
-->
