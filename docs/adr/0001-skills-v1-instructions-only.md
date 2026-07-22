# Skills v1 ships instructions only

The Skills capability discovers Agent Skills and exposes each one's `SKILL.md`
frontmatter (name, description) and body as a deferred Pydantic AI capability.
v1 has no filesystem or command-execution authority: it does not read a skill's
supporting files, does not run bundled scripts, and does not enumerate a skill
directory's contents. Rendered instructions carry the skill body plus the
absolute skill directory (with `${CLAUDE_SKILL_DIR}` expanded) so a portable
body degrades gracefully if the application exposes its own file tools, but
Skills itself grants no access.

An earlier revision of this decision separated skill loading from I/O authority
while still routing to first-party `FileSystem` and `Shell` capabilities and
accepting caller-declared reader and executor tools. That routing is dropped.
Resource and script support returns in a later version once skills sit on a
`ctx.sandbox` abstraction that owns file and command access; the split between
loading and I/O authority is then a property of that abstraction rather than of
Skills.

Unsupported behavioral frontmatter fields are accepted and ignored with one
aggregated warning, keeping portable skills loadable without implying their
invocation, permission, or model-selection behavior is active.
