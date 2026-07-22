# Pydantic AI Harness

Pydantic AI Harness provides optional, reusable agent behaviors built on Pydantic AI primitives.

## Language

**Agent Skill**:
A filesystem package that follows the Agent Skills format, with instructions and metadata in `SKILL.md` and optional bundled resources.
_Avoid_: Programmatic skill

**Capability**:
A reusable unit of agent behavior defined through the Pydantic AI capabilities API. Code-defined instructions and tools are capabilities, not Agent Skills.
_Avoid_: Programmatic skill

**Skill Library**:
A collection of validated Agent Skills made available to an agent as a unit.
_Avoid_: Skill registry
