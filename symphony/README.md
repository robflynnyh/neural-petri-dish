# Symphony Agent Files

This directory contains repo-local Symphony workflow and instruction files.

- `WORKFLOW.md` is the workflow used by the local Symphony service.
- `instructions/` contains binding per-agent instructions loaded by `WORKFLOW.md`.
- Keep local credentials and source overrides in `symphony/.env`; it is ignored by Git.
- Keep temporary agent artifacts under `symphony/.scratch/`, `symphony/tmp/`, or `symphony/logs/`; these are ignored by Git.

Do not commit raw logs, local credentials, or bulky generated artifacts from agent runs.
