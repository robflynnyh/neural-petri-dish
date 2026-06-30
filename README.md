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

Compare mutation dynamics:

```bash
python test_cases/compare_mutation_modes.py --rounds 100 --render-every-rounds 10
```

This writes one video per mutation mode, a per-round CSV with pre-refill survivor counts and previous-round survival rates, and a PNG comparison plot under `test_cases/artifacts/mutation_comparison/`.
Use `--action-mode simultaneous` to batch action proposal inference first, then resolve movement and combat conflicts from the frozen pre-frame grid.

## Notes

- The simulation uses the current terminal size to define the grid.
- Mutated child cells use low-rank structured noise for matrix weights, inspired by EGGROLL-style factorized perturbations.
- `shared_rank1` mutation spawns each new wave from one HP-weighted shared base genome plus per-cell rank-1 perturbations, which is designed to make action inference batchable.
- Saved `*.pkl` state files are ignored by git.
- Dependencies are `numpy`, `torch`, and `sty`.
