# Pydantic AI Harness

Pydantic AI Harness provides optional, reusable agent behaviors built on Pydantic AI primitives.

## Language

**Agent Skill**:
A package in the Agent Skills format. Its `SKILL.md` contains metadata and instructions; the package may also contain bundled files.
_Avoid_: Programmatic skill

**Capability**:
A reusable unit of agent behavior defined through the Pydantic AI capabilities API. Code-defined instructions and tools are capabilities, not Agent Skills.
_Avoid_: Programmatic skill

**Skill Library**:
A directory whose immediate child packages can be loaded by `Skills`.
_Avoid_: Skill registry
