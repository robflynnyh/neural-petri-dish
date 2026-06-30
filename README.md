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

## Notes

- The simulation uses the current terminal size to define the grid.
- Saved `*.pkl` state files are ignored by git.
- Dependencies are `numpy`, `torch`, and `sty`.
