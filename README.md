# Neural Petri Dish

A terminal-based cellular simulation where each cell is controlled by a tiny PyTorch neural network. Cells move, collide, mutate, reproduce, and compete for survival in a grid rendered directly in the terminal.

## Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start a new simulation:

```bash
python neural_petri_dish.py
```

Save state on exit with `Ctrl+C`:

```bash
python neural_petri_dish.py --save state.pkl
```

Resume a saved simulation:

```bash
python neural_petri_dish.py --load state.pkl --save state.pkl
```

## Tests

Install the test dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the test cases:

```bash
pytest test_cases
```

Run a bounded snapshot review:

```bash
python test_cases/vibe_snapshot_review.py
```

This writes text snapshots under `test_cases/artifacts/` and asks `codex exec` to review them against `neural_petri_dish.py`. Use `--no-codex-review` to only collect snapshots, or `--reasoning xhigh` for a heavier Codex review pass.

Render a video artifact for PRs:

```bash
python test_cases/render_video.py --output test_cases/artifacts/neural_petri_dish.mp4
```

The video renderer runs the simulation headlessly and writes an MP4/GIF-style artifact without terminal rendering. Use `--frames`, `--render-rounds`, `--round-stride`, `--fps`, `--size`, `--initial-cells`, and `--seed` to make automated PR previews repeatable.
Use `--render-rounds 5 --round-stride 4` to render every frame inside five full UI rounds sampled as rounds `0,4,8,12,16`; the skipped rounds still run headlessly.

Record shared rank-1 mutation dynamics:

```bash
python test_cases/compare_mutation_modes.py --rounds 100 --render-every-rounds 10
```

This writes a rank-1 run video, a per-round CSV with pre-refill survivor counts and previous-round survival rates, and a PNG plot under `test_cases/artifacts/mutation_comparison/`.

Run the batched GPU mutation harness through the local cooperative GPU scheduler:

```bash
scripts/run_gpu_mutation_benchmark.sh
```

The launcher uses `/store/store5/software/simple-gpu-schedule/with-gpu` and writes JSON/CSV metrics under `test_cases/artifacts/gpu_mutation/`. Override the defaults with environment variables, for example:

```bash
MUTATION_MODE=shared_rank1_factored POPULATION=500000 STEPS=500 scripts/run_gpu_mutation_benchmark.sh --mutation-scale 0.0003
```

For a CPU smoke test of the same code path:

```bash
python scripts/gpu_mutation_benchmark.py --device cpu --population 1000 --steps 10
```

Benchmark end-to-end normal play round time:

```bash
python scripts/benchmark_normal_play.py --action-backend sequential --action-device cpu
/store/store5/software/simple-gpu-schedule/with-gpu any --idle-seconds 5 -- \
  python scripts/benchmark_normal_play.py --engine tensor_rank1 --action-device cuda
```

Run the simulator CLI with the GPU-resident tensor engine:

```bash
/store/store5/software/simple-gpu-schedule/with-gpu any --idle-seconds 5 -- \
  python neural_petri_dish.py --engine tensor_rank1 --action-device cuda
```

Use `--tensor-render-every N` to draw less often without changing the compiled
simulation block size.

Estimate the normal-size round workload on the fast GPU-resident rank-1 tensor
engine. This is a debug projection for the intended simulator engine path; the
current interactive `Game`/`Cell` normal-play loop is still measured with
`benchmark_normal_play.py`.

```bash
/store/store5/software/simple-gpu-schedule/with-gpu any --idle-seconds 5 -- \
  python scripts/benchmark_tensor_normal_rounds.py --device cuda --rounds 3
```

Probe the GPU-resident action kernel, including tensorized neighbor extraction:

```bash
scripts/run_gpu_mutation_benchmark.sh
/store/store5/software/simple-gpu-schedule/with-gpu any --idle-seconds 5 -- \
  python scripts/gpu_action_kernel_benchmark.py --device cuda --cells 200000 --steps 500
/store/store5/software/simple-gpu-schedule/with-gpu any --idle-seconds 5 -- \
  python scripts/gpu_action_kernel_benchmark.py --device cuda --cells 200000 --steps 500 --movement snapshot
/store/store5/software/simple-gpu-schedule/with-gpu any --idle-seconds 5 -- \
  python scripts/gpu_action_kernel_benchmark.py --device cuda --cells 200000 --steps 500 --movement snapshot --families 64
/store/store5/software/simple-gpu-schedule/with-gpu any --idle-seconds 5 -- \
  python scripts/gpu_action_kernel_benchmark.py --device cuda --cells 200000 --steps 500 --movement snapshot_combat --families 64
/store/store5/software/simple-gpu-schedule/with-gpu any --idle-seconds 5 -- \
  python scripts/gpu_action_kernel_benchmark.py --device cuda --cells 200000 --steps 500 --movement snapshot_combat --families 64 --initial-health 15
/store/store5/software/simple-gpu-schedule/with-gpu any --idle-seconds 5 -- \
  python scripts/gpu_action_kernel_benchmark.py --device cuda --cells 2500 --height 60 --width 80 --steps 1500 --movement snapshot_combat --families 1 --initial-health 15 --wave-every 500 --wave-size 300 --wave-initial-health 2
/store/store5/software/simple-gpu-schedule/with-gpu any --idle-seconds 5 -- \
  python scripts/gpu_action_kernel_benchmark.py --device cuda --cells 2500 --height 60 --width 80 --steps 1500 --movement snapshot_combat --families 1 --initial-health 15 --wave-every 500 --wave-size 300 --wave-initial-health 2 --compact-every 250
/store/store5/software/simple-gpu-schedule/with-gpu any --idle-seconds 5 -- \
  python scripts/gpu_action_kernel_benchmark.py --device cuda --cells 2500 --height 60 --width 80 --steps 1500 --movement snapshot_combat --families 1 --initial-health 15 --wave-every 500 --wave-size 300 --wave-initial-health 2 --compact-every 100
/store/store5/software/simple-gpu-schedule/with-gpu any --idle-seconds 5 -- \
  python scripts/gpu_action_kernel_benchmark.py --device cuda --cells 2500 --height 60 --width 80 --steps 1500 --movement snapshot_combat --families 1 --initial-health 15 --wave-every 500 --wave-size 300 --wave-initial-health 2 --compact-every 100 --checksum-actions 0
/store/store5/software/simple-gpu-schedule/with-gpu any --idle-seconds 5 -- \
  python scripts/gpu_action_kernel_benchmark.py --device cuda --cells 2500 --height 60 --width 80 --steps 1500 --movement snapshot_combat --families 1 --initial-health 15 --wave-every 500 --wave-size 300 --wave-initial-health 2 --compact-every 100 --checksum-actions 0 --trace-every 500
/store/store5/software/simple-gpu-schedule/with-gpu any --idle-seconds 5 -- \
  python scripts/gpu_action_kernel_benchmark.py --device cuda --cells 312 --height 60 --width 80 --steps 500 --movement snapshot_combat --families 2 --initial-health 1000 --compact-every 0 --checksum-actions 0 --compiled-step
/store/store5/software/simple-gpu-schedule/with-gpu any --idle-seconds 5 -- \
  python scripts/gpu_action_kernel_benchmark.py --device cuda --cells 2500 --height 60 --width 80 --steps 1500 --movement snapshot_combat --families 1 --initial-health 15 --wave-every 500 --wave-size 300 --wave-initial-health 2 --compact-every 0 --checksum-actions 0 --static-capacity --family-capacity 8 --compiled-step
/store/store5/software/simple-gpu-schedule/with-gpu any --idle-seconds 5 -- \
  python scripts/gpu_action_kernel_benchmark.py --device cuda --cells 2500 --height 60 --width 80 --steps 1500 --warmup-steps 0 --movement snapshot_combat --families 1 --initial-health 15 --health-dtype int32 --matmul-precision high --wave-every 500 --wave-size 300 --wave-initial-health 2 --compact-every 0 --checksum-actions 0 --static-capacity --family-capacity 7 --static-refill-empty --static-refill-check-every 100 --static-rebuild-grid --family-basis-step --compiled-step --compile-mode default --compiled-block-steps 50 --cuda-graph-block
/store/store5/software/simple-gpu-schedule/with-gpu any --idle-seconds 5 -- \
  python scripts/gpu_action_kernel_benchmark.py --device cuda --cells 50000 --height 256 --width 256 --steps 100 --movement snapshot_combat --families 64 --initial-health 1000 --wave-every 25 --wave-size 1000 --wave-initial-health 1000 --compact-every 0 --checksum-actions 0
```

Profile the warmed compiled block for the documented normal-size path:

```bash
/store/store5/software/simple-gpu-schedule/with-gpu any --idle-seconds 5 -- \
  python scripts/profile_gpu_action_kernel.py --device cuda --cells 2500 --height 60 --width 80 --families 1 --family-capacity 7 --initial-health 15 --health-dtype int32 --matmul-precision high --compile-mode default --compiled-block-steps 50 --profile-blocks 1 --output-json test_cases/artifacts/gpu_mutation/profile_block_cap7_block50.json
```

Sweep fixed-capacity compiled settings:

```bash
/store/store5/software/simple-gpu-schedule/with-gpu any --idle-seconds 5 -- \
  python scripts/static_capacity_sweep.py --device cuda --warmup-steps 0 --health-dtypes int32 --matmul-precision high --family-capacities 8,10,16 --refill-check-everys 1,100 --static-rebuild-grid --family-basis-step --compile-mode default --compiled-block-steps 50 --cuda-graph-block --repeats 3 --output-json test_cases/artifacts/gpu_mutation/static_capacity_sweep.json --output-csv test_cases/artifacts/gpu_mutation/static_capacity_sweep.csv
```

## Notes

- The simulation uses the current terminal size to define the grid.
- The active mutation setup is `shared_rank1_factored`. Each spawn wave gets one shared rank-1 family base created from the HP-weighted average of surviving cells; survivors keep their current genome unchanged. Per-cell differences are scalar rank-1 coefficients.
- The factored GPU path applies per-cell rank-1 corrections as `x @ W_base.T + coeff * (x @ v) * u`, so it avoids materializing dense per-cell matrices. Shared-family cells use the same base matrix and rank-1 directions. The CUDA planner packs multiple families into one tensor pass by indexing the right family tensors per cell, which avoids many tiny CUDA launches when later rounds contain small survivor families.
- The default `--action-backend sequential` preserves the current simulation semantics where earlier moves in a frame can affect later cells. `--action-backend family_batched` is an experimental snapshot evaluator for probing GPU batching and should not be treated as the validated exact simulation path.
- Current exact normal play is still fastest on CPU because live cell state is stored as Python objects and the experimental GPU backend rebuilds/transfers batches every frame. The GPU-resident tensor-state benchmark shows the intended direction: keep compact occupancy and index grids, positions, health, states, coefficients, and family tensors resident on GPU, then gather neighbors, evaluate rank-1 actions, apply snapshot movement/combat, compact dead cells, prune unused family tensors, and create new HP-weighted survivor-family waves with tensor operations.
- On an RTX A4500, the normal-size tensor-state probe above is around 1.12s for 1500 steps with `--compact-every 250`, around 1.03-1.14s with `--compact-every 100`, and the larger 50k-cell multi-family combat/wave probe is around 49M processed cell-steps/sec. Use `--checksum-actions 0` for normal-play-style timing without the per-step checksum reduction. Use `--trace-every 500` to time normal-size wave intervals separately; trace segments report both tensor rows and live active cells. The dynamic GPU path shows the sparse-round issue clearly: after the first 500 steps only about 300 cells are active, but later 500-step segments still cost around 0.37s because small-tensor CUDA launch and grid-maintenance overhead dominates the actual rank-1 math.
- `--compiled-step` uses `torch.compile` for fixed-shape CUDA `snapshot_combat` probes and is intended for avoiding sparse-step launch overhead. With `--static-capacity`, the benchmark preallocates fixed cell slots plus `--family-capacity` family rows, keeps dead slots masked by zero health, and inserts waves into inactive slots without changing tensor shapes. This lets the compiled path run across scheduled waves: the fixed-capacity scheduled-wave probe above is about 0.59s compiled versus about 1.12s eager on an RTX A4500. Add `--static-refill-empty` to also refill when the active health mask becomes empty; this is closer to normal play but adds a small synchronization cost. With full-board cell capacity, `--family-capacity 16`, and default int64 health, the refill-enabled compiled probe is about 0.68s with every-step checks and about 0.62s with `--static-refill-check-every 100`; `--health-dtype int32` reduced the 100-step refill-check probe to about 0.56s in repeated runs. `--static-rebuild-grid` rebuilds the playable grid/index area each compiled step instead of using incremental indexed clears; in same-process comparisons on the normal-size int32 static probe it reduced runtime from about 0.56s to about 0.53s. `--family-basis-step` computes base-layer outputs with flattened family-basis matmuls before selecting each cell family; the normal-size int32 rebuild-grid probe now runs around 0.49-0.51s, with traced 500-step segments around 0.17s, 0.17s, and 0.17s. Add `--matmul-precision high` on NVIDIA GPUs to enable TF32-backed float32 matmuls for this benchmark; on the A4500 probe it preserved the same final counts and improved one same-seed run to about 0.49s. `--compile-mode default` was also faster than the previous hard-coded `reduce-overhead` mode for the same family-basis compiled kernel in a same-seed probe, while preserving final counts. `--compiled-block-steps 50` reduces Python dispatch by running 50 compiled fixed-shape steps per call without crossing refill, wave, or trace boundaries. After pre-warming the block graph, dropping the unused final action output, deriving ternary neighbor occupancy from `index_grid`, caching contiguous flattened base-weight matmul views outside the compiled step, and refreshing only changed static-family cache rows, the same normal-size probe ran in about 0.193-0.209s with the same final counts. A traced run with `--trace-every 500` reported segments around 0.067s, 0.062s, and 0.061s, so the later sparse rounds no longer show the old slowdown. A warmed-block profile for the documented cap-7/block-50 path reported about 759us total CUDA time in 100 `aten::mm` calls, 314us in 50 `scatter_reduce_` calls, and 151 launches for one 50-step block; the next meaningful speedup likely needs fewer launches or a fused movement/combat/grid-update kernel rather than another Python-side tweak. Removing the unused binary-grid argument from the family-basis compiled graph preserved counts; a traced normal-size run after that cleanup was about 0.197s. Same-process seed 7/8/9 sweeps found `--family-capacity 7`, `8`, and `9` preserved the same active-cell, refill, and wave counts as `16`; after the boolean neighbor encoding change, `7` had the best average block-50 runtime at about 0.184-0.192s, so the documented normal-size command uses 7. `--family-capacity 6` changed the default-seed dynamics (`waves_spawned=1500`, `active_cells_final=9`), so 7 is the tight lower bound for the current 1500-step normal-size probe. Replacing `sign(index+1)` with boolean comparisons for ternary neighbor encoding preserved counts; a cap-9/block-50 seed 7/8/9 sweep ran about 0.185-0.191s, a small improvement over the clamp/sign variants. Caching the static int32 scatter index vector outside the compiled step preserved counts and improved the cap-7/block-50 seed 7/8/9 sweep to about 0.182-0.186s. Caching a matching all-`-1` vector for dead scatter slots preserved counts but had mixed timing and added another graph argument, so the kernel still uses `torch.full_like(scatter_indices, -1)`. An active-bucket variant processed only padded live-slot buckets while preserving full slot IDs; it matched normal seed-7 final counts but slowed the traced normal-size run to about 0.245s because full-grid scatter/update overhead dominated the smaller matmuls, so it was removed and the documented command keeps the full-capacity block. An occupancy-stamp variant avoided the full index-grid clear in eager CUDA and matched CPU rebuild-grid semantics, but `torch.compile`/Inductor generated an illegal-memory-access Triton kernel during CUDA autotune even on a tiny smoke case, so it is not exposed as a benchmark flag. With `--family-capacity 9`, `--compiled-block-steps 100` preserved the same seed 7/8/9 counts and improved non-traced post-warm loop time to about 0.179-0.189s, but on the current tighter `--family-capacity 7` shape block 100 preserved counts and slowed to about 0.186-0.193s, so the command keeps block 50. Smaller block 25 and mixed block 75 also preserved counts at capacity 7 but did not beat block 50. On the cap-9/block-100 shape, `--compile-mode reduce-overhead` and `max-autotune` both preserved counts but were slower than `default`; `max-autotune` also warned that the A4500 did not have enough CUDA cores for that mode. `--matmul-precision medium` preserved the same counts but was effectively tied with `high` on the A4500 block probes, so the documented command keeps `high`. An `int16` index-grid probe also preserved counts but was tied rather than faster on the normal-size block probes, so the code keeps the simpler int32 index grid. A reusable damage scratch-buffer probe preserved counts but slowed the documented cap-9/block-50 command to about 0.197s, so the compiled kernel keeps `torch.zeros_like(health)` for that temporary. Computing rank-1 scales with small family-basis matmuls for `v_1`/`v_2` preserved counts but slowed cap-9/block-50 runs to about 0.234-0.237s, so the kernel keeps per-cell `v[family]` gathers and elementwise reductions. The CLI defaults `--warmup-steps` to 0 for block mode unless you explicitly ask for benchmark-state warmup before timing; the block graph is already pre-warmed on a cloned state, and the old 20-step warmup would compile an extra partial block. The larger block graph has noticeable compile/warmup cost, so this timing is the post-warmup simulation loop. `--cell-capacity` can be used to sweep smaller slot counts, but tighter capacities can change dynamics: `--cell-capacity 4500` and `4600` both changed refill/wave counts in the normal-size probe. Wider empty-refill checks can also change dynamics; `--static-refill-check-every 200` changed the same-seed final counts, so the documented normal-size command keeps the 100-step check.
- `--cuda-graph-block` captures each warmed compiled fixed-shape block and replays it during timing, which cuts launch overhead without changing the fixed tensor state. The capture pass restores mutable state before timing starts. On the RTX A4500 normal-refill tensor path exposed through `python scripts/benchmark_normal_play.py --engine tensor_rank1 --action-device cuda`, the default block-100 seed-7 run took about 0.0695s total, with 500-step rounds around 0.026s, 0.022s, and 0.021s. That path uses `PER_WAVE=300` and `MIN_WAVE=250`; the same run spawned 1050, 295, and 291 cells across rounds 1-3.
- `--compact-every 0` is useful for high-health, low-death throughput probes where scheduled compaction is mostly wasted work. It is not a drop-in normal-size play setting because deferred dead cells can change empty-board refill timing.
- Saved `*.pkl` state files are ignored by git.
- Dependencies are `numpy`, `torch`, and `sty`.
