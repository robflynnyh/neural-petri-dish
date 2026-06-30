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

## Notes

- The simulation uses the current terminal size to define the grid.
- Saved `*.pkl` state files are ignored by git.
- Dependencies are `numpy`, `torch`, and `sty`.
