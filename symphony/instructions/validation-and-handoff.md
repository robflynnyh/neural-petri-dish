# Validation And Handoff

Run the most targeted command or test that demonstrates the task is complete.

For code changes, prefer targeted pytest coverage first. For tensor-engine
changes, include relevant tests from `test_cases/test_gpu_mutation_benchmark.py`
and then run `PYTHONPATH=. pytest -q` before handoff when feasible.

For render/video changes, validate the generated media with `ffprobe`, inspect
the manifest/metrics, and include the GitHub artifact link in the handoff.

For detached long jobs, smoke test the actual wrapper or callback path before
queueing the long run. A dry run of `scripts/callbacks/linear_experiment_callback.py`
is not a substitute for preserving the wrapper `EXIT` trap.

For documentation-only changes, run `git diff --check` and inspect the diff.

If validation cannot run, document the exact blocker and the command that should
be run later.

Commit completed changes on the issue branch.

Push the branch to `origin`.

Open a GitHub pull request using `/exp/exp4/acp21rjf/scripts/github-create-pr.sh`,
using the issue `Branch/ref` as the PR base when provided, otherwise the
repository default branch.

Include the PR URL in the Linear completion comment.

If pushing or PR creation fails, do not move the issue to `In Review`; post a
blocker comment with the exact failing command and error.

Use the `linear_graphql` tool for Linear updates.

Post one completion comment summarizing files changed, validation, output paths
if any, GitHub PR URL, and residual risk.

Move the issue to `In Review` only when the requested work is complete and the
GitHub handoff has succeeded. Do not move Symphony-completed implementation work
directly to `Done`; leave final acceptance/completion to a human reviewer.

Do not move the issue to `In Review` if the requested work is incomplete,
blocked, not pushed, or missing a PR. In that case, post a blocker comment
explaining exactly what is missing or failing.

Before ending a completed issue, verify with `linear_graphql` that the expected
completion comment exists and that the issue state is `In Review`.
