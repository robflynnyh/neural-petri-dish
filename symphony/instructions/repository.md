# Repository

This repository implements a terminal cellular simulator where cells are driven
by tiny neural networks.

Active implementation focus:
- Prefer the tensor rank-1 engine for performance work.
- Treat legacy dense/low-rank mutation paths as historical unless an issue
  explicitly asks to inspect or remove them.
- The rank-1 path should keep simulation state resident in tensors where
  possible and should exploit shared family base matrices plus rank-1
  per-cell corrections.
- Do not introduce hard gameplay caps on family count. It is acceptable to
  start with a small family capacity when the tensor state can grow and
  recapture as needed.

Important files:
- `neural_petri_dish.py`: interactive CLI and shared gameplay constants.
- `tensor_rank1_sim.py`: GPU-resident tensor simulation state and compiled
  rank-1 action/combat paths.
- `rank1_genome.py`: shared rank-1 genome representation and network sizes.
- `scripts/benchmark_normal_play.py`: normal-play timing path.
- `scripts/benchmark_tensor_normal_rounds.py`: focused tensor-round timing.
- `test_cases/render_tensor_rank1_video.py`: long-run sampled MP4 renderer.
- `test_cases/test_gpu_mutation_benchmark.py`: tensor engine regression tests.
- `test_cases/test_simulation_core.py`: gameplay and core logic tests.
- `scripts/callbacks/linear_experiment_callback.py`: Linear callback helper for
  detached long-running jobs.
- `scripts/templates/queued_experiment_wrapper.template.sh`: Mimas wrapper
  template with an `EXIT` trap that calls the callback helper.
- `scripts/templates/slurm_experiment_wrapper.template.sh`: Stanage wrapper
  template with the same callback discipline.

Generated artifacts:
- `test_cases/artifacts/` is ignored locally. Commit only small artifacts that
  are intentionally useful for review, such as requested MP4s, metrics JSON,
  manifests, and compact final-state debug payloads.
- Keep large scratch, profiler output, and temporary files under ignored paths
  such as `test_cases/artifacts/`, `symphony/.scratch/`, or `symphony/tmp/`.
- Do not use `/tmp` for meaningful scratch work.

Videos and debug runs:
- For requested long-run videos, save the MP4, `.manifest.txt`, `.metrics.json`,
  and final state when useful for debugging.
- Use deterministic seeds and record the exact command and timing.
- If a video shows suspicious behavior, inspect metrics and saved state before
  tuning parameters.

Preserve game semantics when optimizing:
- Normal play is the performance target unless an issue explicitly asks for a
  synthetic benchmark.
- Distinguish sequential interactive behavior from tensor snapshot behavior in
  explanations and validation.
- When changing rules, update both eager and compiled tensor paths, plus tests.
- Food, health, movement, attack, round transition, and spawn rules should stay
  explicit constants in code and visible in metrics/manifests where relevant.
