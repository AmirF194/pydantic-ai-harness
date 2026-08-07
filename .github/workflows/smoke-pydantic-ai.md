---
name: Smoke Pydantic AI
on:
  workflow_dispatch:
permissions:
  contents: read
model: copilot/claude-sonnet-4-5
engine:
  id: pydantic-ai
imports:
  - shared/pydantic-ai-engine.md
network:
  allowed: []
tools:
  bash:
    - "*"
timeout-minutes: 10
---

# Smoke Test: Pydantic AI Engine

Run `git log --oneline -1` and report the resulting commit line. Keep the output to one line.
