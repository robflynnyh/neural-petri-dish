# Experiment Execution

Do not launch long-running GPU work unless the issue asks for a run, render, or
benchmark.

Use `/store/store5/software/simple-gpu-schedule/with-gpu` for cooperative Mimas
GPU allocation instead of manually claiming a GPU. Prefer pool `1,2` unless the
issue or command requires a different pool.

Use Stanage only when the issue or a later human Linear comment directly asks
for Stanage, HPC, or Slurm.

For short GPU checks, run the scheduler-wrapped command directly and report the
exact command and elapsed time.

For long renders or benchmarks, prefer durable detached `screen` sessions with
logs under `symphony/logs/` or `test_cases/artifacts/`. Record:
- command
- branch and commit
- log path
- output artifact paths
- expected completion check

Do not spend agent turns waiting for long queued or running work unless the
issue explicitly asks for an inline result. If the job will outlive the turn,
post a Linear queue comment and move the issue back to `Backlog`.

Before claiming a speedup, compare normal-play timing where possible. Synthetic
microbenchmarks are useful for diagnosis but should not replace normal-play
measurements.
