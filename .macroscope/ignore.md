# Repository-wide Macroscope ignore (code review + any check-run agents).
#
# One glob per line; `#` starts a comment; `**` spans directories. Files listed
# here are dropped from the billed review diff, so Macroscope does not spend
# credits reviewing recorded or mechanical files that no human reads.

# Lock files: uv.lock diffs are large and mechanical on dependency updates.
**/*.lock

# Recorded VCR cassettes (regenerated, not hand-reviewed).
**/cassettes/**

# Documentation images (binary, not reviewable as a diff).
docs/images/**
