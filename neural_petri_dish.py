import numpy as np
import torch
import argparse
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import sys
import time

from rank1_genome import (
    ATTACK_OUTPUT_INDEX,
    DIRECTION_OUTPUT_DIM,
    FACTORED_WAVE_COEFF_SCALE,
    HIDDEN_DIM,
    LinearGenes,
    NEIGHBOR_INPUT_DIM,
    NETWORK_INPUT_DIM,
    OUTPUT_DIM,
    SharedRank1Family,
    factored_gene_batch,
    factored_gene_tensors,
    factored_genes,
)

try:
    from sty import fg, bg
except ModuleNotFoundError:
    class _NoStyle:
        rs = ''

        def __call__(self, *_args, **_kwargs):
            return ''

        def __getattr__(self, _name):
            return ''

    fg = _NoStyle()
    bg = _NoStyle()


def cls():
    if os.getenv('TERM'):
        sys.stdout.write('\033[2J\033[H')
        sys.stdout.flush()


def hide_cursor():
    if os.getenv('TERM'):
        sys.stdout.write('\033[?25l')
        sys.stdout.flush()


def show_cursor():
    if os.getenv('TERM'):
        sys.stdout.write('\033[?25h')
        sys.stdout.flush()


def terminal_size(size=None):
    if size is None:
        try:
            return os.get_terminal_size()
        except OSError:
            return shutil.get_terminal_size(fallback=(80, 24))
    if hasattr(size, 'lines') and hasattr(size, 'columns'):
        return size
    lines, columns = size
    return os.terminal_size((columns, lines))


def parse_size(size):
    if size is None:
        return None
    normalized = size.lower().replace(',', 'x')
    try:
        lines, columns = normalized.split('x', 1)
        return int(lines), int(columns)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('size must use LINESxCOLUMNS, for example 24x80') from exc


def positive_int(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('value must be an integer') from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError('value must be positive')
    return parsed


def seed_all(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def empty_positions(game):
    play_area = game.grid[2:game.size.lines, 2:game.size.columns + 2]
    positions = np.argwhere(play_area == 0)
    return positions + np.array([2, 2])


col = 16
X = f'{bg(col)}❏{bg.rs}'# the icon for the cell *·◉ ○ ●○○✺✺
BLANK = f'{bg(col)} {bg.rs}' # icon for empty cell °
FRAME_RATE = 0.05    # seconds between frames
MUTATION_MODE_SHARED_RANK1_FACTORED = 'shared_rank1_factored'
MUTATION_MODES = (MUTATION_MODE_SHARED_RANK1_FACTORED,)
DEFAULT_MUTATION_MODE = MUTATION_MODE_SHARED_RANK1_FACTORED
ACTION_BACKEND_SEQUENTIAL = 'sequential'
ACTION_BACKEND_FAMILY_BATCHED = 'family_batched'
ACTION_BACKENDS = (ACTION_BACKEND_SEQUENTIAL, ACTION_BACKEND_FAMILY_BATCHED)
SIM_ENGINE_GAME = 'game'
SIM_ENGINE_TENSOR_RANK1 = 'tensor_rank1'
SIM_ENGINES = (SIM_ENGINE_GAME, SIM_ENGINE_TENSOR_RANK1)
DIRECTION_DELTAS = (
    (0, 0),   # stationary
    (-1, 0),  # up
    (1, 0),   # down
    (0, 1),   # right
    (0, -1),  # left
    (-1, 1),  # up right
    (-1, -1), # up left
    (1, 1),   # down right
    (1, -1),  # down left
)
BASE_ATTACK_DAMAGE = 1
LONE_TARGET_DAMAGE_BONUS = 1
FOOD_PER_ROUND = 25
FOOD_HEALTH_REWARD = 2.5
FOOD_INPUT_VALUE = 2
NPC_COUNT = 50
NPC_INPUT_VALUE = 3
NEIGHBOR_OFFSETS = np.array(
    [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if not (dy == 0 and dx == 0)],
    dtype=np.int64,
)

direction_dict = {
    0: lambda yx: np.array(yx), #stationary
    1: lambda yx: np.array([yx[0]-1, yx[1]]), # up
    2: lambda yx: np.array([yx[0]+1, yx[1]]), # down
    3: lambda yx: np.array([yx[0], yx[1]+1]), # right
    4: lambda yx: np.array([yx[0], yx[1]-1]), # left
    5: lambda yx: np.array([yx[0]-1, yx[1]+1]), # up right
    6: lambda yx: np.array([yx[0]-1, yx[1]-1]), # up left
    7: lambda yx: np.array([yx[0]+1, yx[1]+1]), # down right
    8: lambda yx: np.array([yx[0]+1, yx[1]-1]) # down left
}


def target_has_other_immediate_neighbors(game, target_y, target_x, attacker):
    index = game.cells_by_pos
    stride = game._cell_key_stride
    for dy, dx in DIRECTION_DELTAS[1:]:
        neighbor = index.get((target_y + dy) * stride + target_x + dx)
        if neighbor is not None and neighbor is not attacker:
            return True
    return False


def attack_damage_for_target(game, target_y, target_x, attacker):
    damage = BASE_ATTACK_DAMAGE
    if not target_has_other_immediate_neighbors(game, target_y, target_x, attacker):
        damage += LONE_TARGET_DAMAGE_BONUS
    return damage


def pack_action(direction, attack):
    return int(direction) + (DIRECTION_OUTPUT_DIM if attack else 0)


def action_direction(action):
    return int(action) % DIRECTION_OUTPUT_DIM


def action_is_attack(action):
    return int(action) >= DIRECTION_OUTPUT_DIM


def action_from_logits(logits):
    direction = int(logits[:DIRECTION_OUTPUT_DIM].argmax())
    attack = bool(logits[ATTACK_OUTPUT_INDEX] > 0.0)
    return pack_action(direction, attack)

class Cell:
    '''
    Manages each cell, it's genes and it's neural network
    '''
    def __init__(self, pos, genes=None, initialize_genes=True):
        if genes is None and initialize_genes:
            self.linear = LinearGenes(NETWORK_INPUT_DIM, HIDDEN_DIM)
            self.linear2 = LinearGenes(HIDDEN_DIM, OUTPUT_DIM)
        elif genes is not None:
            self.linear = LinearGenes(
                weight=genes['weight_1'],
                bias=genes['bias_1'],
                clone_weight=genes.get('_clone_weight_1', True),
                clone_bias=genes.get('_clone_bias_1', True),
            )
            self.linear2 = LinearGenes(
                weight=genes['weight_2'],
                bias=genes['bias_2'],
                clone_weight=genes.get('_clone_weight_2', True),
                clone_bias=genes.get('_clone_bias_2', True),
            )
        else:
            self.linear = None
            self.linear2 = None
        self.input_buffer = np.zeros(NETWORK_INPUT_DIM, dtype=np.float32)
        self.hidden_buffer = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.output_buffer = np.zeros(OUTPUT_DIM, dtype=np.float32)
        self.y = int(pos[0])
        self.x = int(pos[1])
        self.pos = [self.y, self.x]
        self.health = 2.0
        self.max_health = 15.0
        self.age = 0
        self.prev_state = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.diversity = None
        self.rank1_family = None
        self.rank1_coeff_1 = 0.0
        self.rank1_coeff_2 = 0.0

        self.stationary_steps = 0

        if genes is not None:
            self.rank1_family = genes.get('_rank1_family')
            self.rank1_coeff_1 = float(genes.get('_rank1_coeff_1', 0.0))
            self.rank1_coeff_2 = float(genes.get('_rank1_coeff_2', 0.0))

    def forward(self, neighbors):
        if isinstance(neighbors, torch.Tensor):
            neighbors = neighbors.detach().cpu().numpy()
        return self.forward_neighbors(neighbors)

    def forward_neighbors(self, neighbors):
        neighbors = neighbors.reshape(-1)
        buffer = self.input_buffer
        if neighbors.shape[0] == 25:
            buffer[:12] = neighbors[:12]
            buffer[12:NEIGHBOR_INPUT_DIM] = neighbors[13:]
        else:
            buffer[:NEIGHBOR_INPUT_DIM] = neighbors
        return self.forward_from_input_buffer()

    def forward_neighbors25(self, neighbors):
        flat = neighbors.reshape(25)
        return self.forward_flat_neighbors25(flat)

    def forward_flat_neighbors25(self, flat):
        buffer = self.input_buffer
        buffer[:12] = flat[:12]
        buffer[12:NEIGHBOR_INPUT_DIM] = flat[13:]
        buffer[NEIGHBOR_INPUT_DIM:] = self.prev_state
        hidden = self.hidden_buffer
        output = self.output_buffer
        np.dot(self.linear.weight_np, buffer, out=hidden)
        hidden += self.linear.bias_np
        np.maximum(hidden, 0, out=hidden)
        self.commit_forward_state(hidden)
        np.dot(self.linear2.weight_np, hidden, out=output)
        output += self.linear2.bias_np
        return action_from_logits(output)

    def forward_from_input_buffer(self):
        buffer = self.input_buffer
        buffer[NEIGHBOR_INPUT_DIM:] = self.prev_state
        hidden = self.hidden_buffer
        output = self.output_buffer
        np.dot(self.linear.weight_np, buffer, out=hidden)
        hidden += self.linear.bias_np
        np.maximum(hidden, 0, out=hidden)
        self.commit_forward_state(hidden)
        np.dot(self.linear2.weight_np, hidden, out=output)
        output += self.linear2.bias_np
        return action_from_logits(output)

    def commit_forward_state(self, hidden):
        self.prev_state[:] = hidden

    def update_pos(self, pos):
        self.y = int(pos[0])
        self.x = int(pos[1])
        self.pos = [self.y, self.x]
                    
            

    def get_genes(self):
        return {
            'weight_1': self.linear.weight, 
            'bias_1': self.linear.bias,
            'weight_2': self.linear2.weight,
            'bias_2': self.linear2.bias
            }

    def add_health(self, amount=1):
        self.health = min(self.max_health, self.health + amount)

    def total_parameters(self):
        return (
            self.linear.weight.numel()
            + self.linear.bias.numel()
            + self.linear2.weight.numel()
            + self.linear2.bias.numel()
        )


def random_spawn(game, check_available=True):
    '''returns a random position in the grid that is not occupied'''
    if check_available and len(empty_positions(game)) == 0:
        raise RuntimeError('No Empty Positions Available')
    # Keep the original rejection-sampling draw order so seeded runs match the
    # interactive simulation as closely as possible. The pre-check only prevents
    # an infinite loop when the grid is full.
    while True:
        pos = np.array([np.random.randint(2, game.size.lines), np.random.randint(2, game.size.columns+2)])
        if game.grid[pos[0], pos[1]] == 0:
            return pos


class Game():
    '''
    Manages Game State
    '''
    def __init__(
            self,
            genepool=None,
            size=None,
            mutation_mode=DEFAULT_MUTATION_MODE,
            action_backend=ACTION_BACKEND_SEQUENTIAL,
            action_device='auto',
            batched_min_family_size=32):
        self.size = terminal_size(size)
        self.grid = np.zeros((self.size.lines+2, self.size.columns+4), dtype=np.float32)
        # set the 2 layer border as -1's (each cell has a vision of 4x4 hence border to avoid out of bounds)
        self.grid[:, 0:2] = -1
        self.grid[:, -2:] = -1
        self.grid[0:2, :] = -1
        self.grid[-2:, :] = -1
        self._cell_key_stride = self.grid.shape[1]
        self.rounds = 0
        #
        self.cells = []
        self.cells_by_pos = {}
        #self.graveyard = []
        self.mutate_rate = 0.00001
        self.mutation_mode = mutation_mode
        self.action_backend = action_backend
        self.action_device = action_device
        self.batched_min_family_size = batched_min_family_size
        self.defer_cell_list_removals = False
        self.cells_removed_this_step = False

    def _cell_key(self, y, x):
        return int(y) * self._cell_key_stride + int(x)

    def _rebuild_cell_index(self):
        self.cells_by_pos = {self._cell_key(cell.y, cell.x): cell for cell in self.cells}
        return self.cells_by_pos

    def _cell_index(self):
        if not hasattr(self, 'cells_by_pos'):
            return self._rebuild_cell_index()
        return self.cells_by_pos

    def get_cell(self, y, x):
        return self.cells_by_pos.get(int(y) * self._cell_key_stride + int(x), False)

    def update_cell(self, y, x, new, cell=None):
        new_y = int(new[0])
        new_x = int(new[1])
        new = [new_y, new_x]
        if cell is None:
            cell = self.get_cell(y, x)
            if cell == False:
                raise Exception('Cell Does not Exist at this Position')
        occupant = self.get_cell(new_y, new_x)
        if occupant is not False and occupant is not cell:
            raise Exception('New Position is Occupied')
        if self.grid[new_y, new_x] == -1:
            raise Exception('New Position is Outside the Play Area')
       
        old_key = int(y) * self._cell_key_stride + int(x)
        new_key = new_y * self._cell_key_stride + new_x
        cell.update_pos(new)
        self.grid[new_y, new_x] = 1
        index = self.cells_by_pos
        index.pop(old_key, None)
        index[new_key] = cell
        if [y, x] != new:
            self.grid[y][x] = 0

    def remove_cell(self, y, x): 
        #self.graveyard.append(self.get_cell(y, x)) # change so cell is passed in
        self.grid[y, x] = 0
        cell = self.cells_by_pos.pop(int(y) * self._cell_key_stride + int(x), None)
        if self.defer_cell_list_removals:
            self.cells_removed_this_step = True
            return
        if cell is not None:
            try:
                self.cells.remove(cell)
            except ValueError:
                self._rebuild_cell_index()

    def compact_cells(self):
        index = self.cells_by_pos
        stride = self._cell_key_stride
        self.cells = [cell for cell in self.cells if index.get(cell.y * stride + cell.x) is cell]

    def damage_cell(self, cell, amount=1):
        if cell == False or self.get_cell(*cell.pos) is not cell:
            return False
        cell.health -= amount
        if cell.health <= 0:
            self.remove_cell(*cell.pos)
            return True
        return False

    def apply_round_transition_health_cost(self):
        if ROUND_TRANSITION_HEALTH_COST <= 0:
            return
        for cell in list(self.cells):
            self.damage_cell(cell, ROUND_TRANSITION_HEALTH_COST)
        if self.cells_removed_this_step:
            self.compact_cells()
            self.cells_removed_this_step = False

    def add_cell(self, y, x, genes=None):
        if self.grid[y][x] != 0:
            raise Exception('Cannot Add Cell to a Non-Empty Position')
        self.grid[y][x] = 1
        cell = Cell([y, x], genes)
        self.cells.append(cell)
        self.cells_by_pos[int(y) * self._cell_key_stride + int(x)] = cell

    def add_factored_cell(self, y, x, family, coeff_1, coeff_2, weight_1, weight_2):
        if self.grid[y][x] != 0:
            raise Exception('Cannot Add Cell to a Non-Empty Position')
        self.grid[y][x] = 1
        cell = Cell([y, x], initialize_genes=False)
        cell.linear = LinearGenes(weight=weight_1, bias=family.base_bias_1, clone_weight=False, clone_bias=False)
        cell.linear2 = LinearGenes(weight=weight_2, bias=family.base_bias_2, clone_weight=False, clone_bias=False)
        cell.rank1_family = family
        cell.rank1_coeff_1 = float(coeff_1)
        cell.rank1_coeff_2 = float(coeff_2)
        self.cells.append(cell)
        self.cells_by_pos[int(y) * self._cell_key_stride + int(x)] = cell

    def make_factored_wave_family(self):
        if len(self.cells) == 0:
            return SharedRank1Family()
        health = torch.tensor([max(cell.health, 0) for cell in self.cells], dtype=torch.float32)
        if health.sum() <= 0:
            health.fill_(1.0)
        weights = health / health.sum()
        weighted_genes = {}
        for key in ('weight_1', 'bias_1', 'weight_2', 'bias_2'):
            tensors = torch.stack([cell.get_genes()[key] for cell in self.cells])
            view_shape = (len(self.cells),) + (1,) * (tensors.ndim - 1)
            weighted_genes[key] = (tensors * weights.reshape(view_shape)).sum(dim=0)
        return SharedRank1Family(weighted_genes)

    def wave_factored_gene_batch(self, family, count):
        return factored_gene_batch(family, count)

    def wave_factored_tensors(self, family, count):
        return factored_gene_tensors(family, count)

    def mutate_factored(self, cell):
        family = cell.rank1_family
        if family is None:
            family = SharedRank1Family(cell.get_genes())
            cell.rank1_family = family
            cell.rank1_coeff_1 = 0.0
            cell.rank1_coeff_2 = 0.0

        if np.random.rand() < 0.05:
            y, x = cell.y, cell.x
            ncells = []
            for dy, dx in DIRECTION_DELTAS[1:]:
                ncell = self.get_cell(y + dy, x + dx)
                if ncell != False:
                    ncells.append(ncell)
            if len(ncells) != 0:
                genes = [c.get_genes() for c in ncells]
                new_genes = {}
                for key in ('weight_1', 'bias_1', 'weight_2', 'bias_2'):
                    new_genes[key] = torch.mean(torch.stack([g[key] for g in genes]), dim=0)
                blended = {
                    'weight_1': cell.linear.weight * 0.8 + new_genes['weight_1'] * 0.2,
                    'bias_1': cell.linear.bias * 0.8 + new_genes['bias_1'] * 0.2,
                    'weight_2': cell.linear2.weight * 0.8 + new_genes['weight_2'] * 0.2,
                    'bias_2': cell.linear2.bias * 0.8 + new_genes['bias_2'] * 0.2,
                }
                new_family = SharedRank1Family(blended)
                return factored_genes(
                    new_family,
                    np.random.randn() * 0.1,
                    np.random.randn() * 0.1,
                    blended['bias_1'],
                    blended['bias_2'],
                )
            return factored_genes(
                family,
                cell.rank1_coeff_1 + np.random.randn() * 0.001,
                cell.rank1_coeff_2 + np.random.randn() * 0.001,
                cell.linear.bias + torch.randn_like(cell.linear.bias) * 0.001,
                cell.linear2.bias + torch.randn_like(cell.linear2.bias) * 0.001,
            )
        elif np.random.rand() < 0.4:
            return factored_genes(
                family,
                cell.rank1_coeff_1 + np.random.randn() * self.mutate_rate,
                cell.rank1_coeff_2 + np.random.randn() * self.mutate_rate,
                cell.linear.bias + torch.randn_like(cell.linear.bias) * self.mutate_rate,
                cell.linear2.bias + torch.randn_like(cell.linear2.bias) * self.mutate_rate,
            )
        return factored_genes(
            family,
            cell.rank1_coeff_1,
            cell.rank1_coeff_2,
            cell.linear.bias,
            cell.linear2.bias,
        )

    def mutate(self, cell):
        '''
        Mutates a cell's genes using the shared rank-1 family representation.
        '''
        return self.mutate_factored(cell)


def render_grid(game, styled=True):
    # skip the 2 layer border
    rows = []
    for row in range(2, game.size.lines):
        pstring = [BLANK if styled else ' ']*game.size.columns
        cells_row = game.grid[row]
        #if sum(cells_row) != 0: # doesn't account for padding
        for col in range(2, game.size.columns + 2):
            if cells_row[col] == 1:
                color = int((87 - 4*np.sum(game.grid[row-1:row+2,col-1:col+2])) % 255) # color is based on density of cells # could remove -1's to account for border padding
                pstring[col - 2] = f'{fg(color)}{X}{fg.rs}' if styled else '#'
        rows.append(''.join(pstring))
    return '\n'.join(rows)


def render_tensor_grid(index_grid, size, styled=True):
    rows = []
    for row in range(2, size.lines):
        pstring = [BLANK if styled else ' '] * size.columns
        index_row = index_grid[row]
        for col in range(2, size.columns + 2):
            if index_row[col] >= 0:
                color = int((87 - 4 * np.sum(index_grid[row-1:row+2, col-1:col+2] >= 0)) % 255)
                pstring[col - 2] = f'{fg(color)}{X}{fg.rs}' if styled else '#'
        rows.append(''.join(pstring))
    return '\n'.join(rows)


def print_grid(game):
    print(render_grid(game))


def render_frame(game, countdown):
    return f'{render_grid(game)}\n{status_line(game, countdown)}'


def draw_frame(game, countdown, first_frame=False):
    # Redraw in-place instead of shelling out to clear. One buffered write per
    # frame prevents most flicker while still drawing every simulation frame.
    if first_frame:
        cls()
    elif os.getenv('TERM'):
        sys.stdout.write('\033[H')
    sys.stdout.write(render_frame(game, countdown))
    sys.stdout.write('\033[J\n')
    sys.stdout.flush()


def status_line(game, countdown, styled=True):
    totalcells = len(game.cells)
    if totalcells == 0:
        avghealth = 0
        maxhealth = 0
        maxage = 0
    else:
        avghealth = round(sum([c.health for c in game.cells]) / totalcells)
        maxhealth = max([c.health for c in game.cells])
        maxage = max([c.age for c in game.cells])

    if not styled:
        return (
            f'Petri Dish      AVG HP: {avghealth}   MAX HP: {maxhealth}    '
            f'Total Players: {totalcells}   Oldest Cell: {maxage}   '
            f'Total Rounds: {game.rounds}    Countdown: {countdown}'
        )

    return (
        f'{fg.red}Petri Dish{fg.rs}{fg.green}      AVG HP: {avghealth}   '
        f'MAX HP: {maxhealth}{fg.rs}    {fg.white}Total Players: {totalcells}'
        f'{fg.rs}   {fg.red}Oldest Cell: {maxage}   Total Rounds: {game.rounds}'
        f'{fg.rs}    {fg.blue}Countdown: {countdown}{fg.rs}'
    )


def tensor_status_line(health, rounds, countdown, styled=True):
    active_health = health[health > 0]
    totalcells = int(active_health.shape[0])
    if totalcells == 0:
        avghealth = 0
        maxhealth = 0
    else:
        avghealth = round(float(active_health.mean()))
        maxhealth = int(active_health.max())

    if not styled:
        return (
            f'Petri Dish      AVG HP: {avghealth}   MAX HP: {maxhealth}    '
            f'Total Players: {totalcells}   Oldest Cell: 0   '
            f'Total Rounds: {rounds}    Countdown: {countdown}'
        )

    return (
        f'{fg.red}Petri Dish{fg.rs}{fg.green}      AVG HP: {avghealth}   '
        f'MAX HP: {maxhealth}{fg.rs}    {fg.white}Total Players: {totalcells}'
        f'{fg.rs}   {fg.red}Oldest Cell: 0   Total Rounds: {rounds}'
        f'{fg.rs}    {fg.blue}Countdown: {countdown}{fg.rs}'
    )


def write_snapshot(game, countdown, frame, snapshot_dir):
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = [
        f'Frame: {frame}',
        status_line(game, countdown, styled=False),
        render_grid(game, styled=False),
        '',
    ]
    path = snapshot_dir / f'frame_{frame:05d}.txt'
    path.write_text('\n'.join(snapshot), encoding='utf-8')
    return path


def tensor_frame_text(index_grid, health, size, rounds, countdown, styled=True):
    return (
        f'{render_tensor_grid(index_grid, size, styled=styled)}\n'
        f'{tensor_status_line(health, rounds, countdown, styled=styled)}'
    )


def draw_tensor_frame(index_grid, health, size, rounds, countdown, first_frame=False):
    if first_frame:
        cls()
    elif os.getenv('TERM'):
        sys.stdout.write('\033[H')
    sys.stdout.write(tensor_frame_text(index_grid, health, size, rounds, countdown, styled=True))
    sys.stdout.write('\033[J\n')
    sys.stdout.flush()


def write_tensor_snapshot(index_grid, health, size, rounds, countdown, frame, snapshot_dir):
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = [
        f'Frame: {frame}',
        tensor_status_line(health, rounds, countdown, styled=False),
        render_tensor_grid(index_grid, size, styled=False),
        '',
    ]
    path = snapshot_dir / f'frame_{frame:05d}.txt'
    path.write_text('\n'.join(snapshot), encoding='utf-8')
    return path


def advance_round(game, countdown):
    if countdown == 0:
        game.apply_round_transition_health_cost()
        totalcells = len(game.cells)
        maxage = max([c.age for c in game.cells]) if game.cells else 0
        game = init(game, num=max(PER_WAVE - totalcells, MIN_WAVE))
        countdown = ROUNDTIME
        game.rounds += 1

        if totalcells > MAX_TOTAL:
            game = prune(game) # prune the game if it is too big
        for cell in game.cells:
            cell.age += 1
            if cell.age == maxage:
                cell.add_health()
    return game, countdown


def resolve_action_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def planned_packed_family_action_list(game, groups, device, cell_count):
    families = list(groups)
    packed_cells = []
    planned_indices = []
    family_indices = []
    for family_index, family in enumerate(families):
        group = groups[family]
        for planned_index, cell in group:
            packed_cells.append(cell)
            planned_indices.append(planned_index)
            family_indices.append(family_index)

    if not packed_cells:
        return None

    inputs = np.empty((len(packed_cells), NETWORK_INPUT_DIM), dtype=np.float32)
    positions = np.asarray([(cell.y, cell.x) for cell in packed_cells], dtype=np.int64)
    inputs[:, :NEIGHBOR_INPUT_DIM] = game.grid[
        positions[:, 0, None] + NEIGHBOR_OFFSETS[None, :, 0],
        positions[:, 1, None] + NEIGHBOR_OFFSETS[None, :, 1],
    ]
    inputs[:, NEIGHBOR_INPUT_DIM:] = np.asarray([cell.prev_state for cell in packed_cells], dtype=np.float32)
    coeff_1 = np.asarray([cell.rank1_coeff_1 for cell in packed_cells], dtype=np.float32)
    coeff_2 = np.asarray([cell.rank1_coeff_2 for cell in packed_cells], dtype=np.float32)
    bias_1 = np.asarray([cell.linear.bias_np for cell in packed_cells], dtype=np.float32)
    bias_2 = np.asarray([cell.linear2.bias_np for cell in packed_cells], dtype=np.float32)

    cached = [family.tensors(device) for family in families]
    base_weight_1 = torch.stack([item['base_weight_1'] for item in cached])
    base_weight_2 = torch.stack([item['base_weight_2'] for item in cached])
    u_1 = torch.stack([item['u_1'] for item in cached])
    v_1 = torch.stack([item['v_1'] for item in cached])
    u_2 = torch.stack([item['u_2'] for item in cached])
    v_2 = torch.stack([item['v_2'] for item in cached])

    input_tensor = torch.as_tensor(inputs, device=device)
    family_index_tensor = torch.as_tensor(family_indices, device=device, dtype=torch.long)
    coeff_1_tensor = torch.as_tensor(coeff_1, device=device)
    coeff_2_tensor = torch.as_tensor(coeff_2, device=device)
    bias_1_tensor = torch.as_tensor(bias_1, device=device)
    bias_2_tensor = torch.as_tensor(bias_2, device=device)

    if len(families) == 1:
        base_hidden = input_tensor.matmul(base_weight_1[0].t())
        rank1_hidden = (
            input_tensor.matmul(v_1[0]) * coeff_1_tensor
        ).unsqueeze(1) * u_1[0].unsqueeze(0)
        hidden = base_hidden.add_(rank1_hidden).add_(bias_1_tensor).relu_()
        base_logits = hidden.matmul(base_weight_2[0].t())
        rank1_logits = (
            hidden.matmul(v_2[0]) * coeff_2_tensor
        ).unsqueeze(1) * u_2[0].unsqueeze(0)
    else:
        selected_base_weight_1 = base_weight_1[family_index_tensor]
        selected_v_1 = v_1[family_index_tensor]
        selected_u_1 = u_1[family_index_tensor]
        base_hidden = torch.bmm(selected_base_weight_1, input_tensor.unsqueeze(2)).squeeze(2)
        rank1_hidden = (
            (input_tensor * selected_v_1).sum(dim=1) * coeff_1_tensor
        ).unsqueeze(1) * selected_u_1
        hidden = base_hidden.add_(rank1_hidden).add_(bias_1_tensor).relu_()

        selected_base_weight_2 = base_weight_2[family_index_tensor]
        selected_v_2 = v_2[family_index_tensor]
        selected_u_2 = u_2[family_index_tensor]
        base_logits = torch.bmm(selected_base_weight_2, hidden.unsqueeze(2)).squeeze(2)
        rank1_logits = (
            (hidden * selected_v_2).sum(dim=1) * coeff_2_tensor
        ).unsqueeze(1) * selected_u_2
    logits = base_logits.add_(rank1_logits).add_(bias_2_tensor)

    directions = logits[:, :DIRECTION_OUTPUT_DIM].argmax(dim=1)
    attacks = logits[:, ATTACK_OUTPUT_INDEX] > 0
    actions = (directions + attacks.to(directions.dtype) * DIRECTION_OUTPUT_DIM).detach().cpu().numpy()
    hidden_np = hidden.detach().cpu().numpy()
    planned = [None] * cell_count
    for index, planned_index in enumerate(planned_indices):
        planned[planned_index] = (int(actions[index]), hidden_np[index])
    return planned


def planned_grouped_family_action_list(game, groups, device, cell_count):
    planned = [None] * cell_count
    for family, group in groups.items():
        group_cells = [cell for _planned_index, cell in group]
        inputs = np.empty((len(group_cells), NETWORK_INPUT_DIM), dtype=np.float32)
        positions = np.asarray([(cell.y, cell.x) for cell in group_cells], dtype=np.int64)
        inputs[:, :NEIGHBOR_INPUT_DIM] = game.grid[
            positions[:, 0, None] + NEIGHBOR_OFFSETS[None, :, 0],
            positions[:, 1, None] + NEIGHBOR_OFFSETS[None, :, 1],
        ]
        inputs[:, NEIGHBOR_INPUT_DIM:] = np.asarray([cell.prev_state for cell in group_cells], dtype=np.float32)
        coeff_1 = np.asarray([cell.rank1_coeff_1 for cell in group_cells], dtype=np.float32)
        coeff_2 = np.asarray([cell.rank1_coeff_2 for cell in group_cells], dtype=np.float32)
        bias_1 = np.asarray([cell.linear.bias_np for cell in group_cells], dtype=np.float32)
        bias_2 = np.asarray([cell.linear2.bias_np for cell in group_cells], dtype=np.float32)

        family_tensors = family.tensors(device)
        input_tensor = torch.as_tensor(inputs, device=device)
        coeff_1_tensor = torch.as_tensor(coeff_1, device=device)
        coeff_2_tensor = torch.as_tensor(coeff_2, device=device)
        bias_1_tensor = torch.as_tensor(bias_1, device=device)
        bias_2_tensor = torch.as_tensor(bias_2, device=device)

        base_hidden = input_tensor.matmul(family_tensors['base_weight_1'].t())
        rank1_hidden = (
            input_tensor.matmul(family_tensors['v_1']) * coeff_1_tensor
        ).unsqueeze(1) * family_tensors['u_1'].unsqueeze(0)
        hidden = base_hidden.add_(rank1_hidden).add_(bias_1_tensor).relu_()
        base_logits = hidden.matmul(family_tensors['base_weight_2'].t())
        rank1_logits = (
            hidden.matmul(family_tensors['v_2']) * coeff_2_tensor
        ).unsqueeze(1) * family_tensors['u_2'].unsqueeze(0)
        logits = base_logits.add_(rank1_logits).add_(bias_2_tensor)

        directions = logits[:, :DIRECTION_OUTPUT_DIM].argmax(dim=1)
        attacks = logits[:, ATTACK_OUTPUT_INDEX] > 0
        actions = (directions + attacks.to(directions.dtype) * DIRECTION_OUTPUT_DIM).detach().cpu().numpy()
        hidden_np = hidden.detach().cpu().numpy()
        for index, (planned_index, _cell) in enumerate(group):
            planned[planned_index] = (int(actions[index]), hidden_np[index])
    return planned


def planned_family_action_list(game, cells):
    device = resolve_action_device(getattr(game, 'action_device', 'auto'))
    min_family_size = getattr(game, 'batched_min_family_size', 32)
    groups = {}
    for planned_index, cell in enumerate(cells):
        family = getattr(cell, 'rank1_family', None)
        if family is None:
            continue
        groups.setdefault(family, []).append((planned_index, cell))

    groups = {family: group for family, group in groups.items() if len(group) >= min_family_size}
    if not groups:
        return None
    if device.type == 'cuda':
        return planned_packed_family_action_list(game, groups, device, len(cells))
    return planned_grouped_family_action_list(game, groups, device, len(cells))


def planned_family_actions(game, cells):
    planned_list = planned_family_action_list(game, cells)
    if planned_list is None:
        return {}
    return {
        id(cell): planned
        for cell, planned in zip(cells, planned_list)
        if planned is not None
    }


def apply_stationary_health_cost(game, cell, old_y, old_x):
    if STATIONARY_HEALTH_COST <= 0:
        return
    stride = game._cell_key_stride
    if game.cells_by_pos.get(cell.y * stride + cell.x) is not cell:
        return
    if cell.y == old_y and cell.x == old_x:
        cell.stationary_steps += 1
    else:
        cell.stationary_steps = 0
    if cell.stationary_steps >= STATIONARY_DAMAGE_AFTER_STEPS:
        game.damage_cell(cell, STATIONARY_HEALTH_COST)


def apply_cell_action(game, cell, action):
    direction = action_direction(action)
    grid = game.grid
    index = game.cells_by_pos
    stride = game._cell_key_stride
    y, x = cell.y, cell.x
    if direction == 0:
        apply_stationary_health_cost(game, cell, y, x)
        return

    attack_intent = action_is_attack(action)
    dy, dx = DIRECTION_DELTAS[direction]
    new_y = y + dy
    new_x = x + dx
    target_value = grid[new_y, new_x]

    if target_value == -1:
        if not attack_intent:
            cell.health -= 1
            if cell.health <= 0:
                grid[y, x] = 0
                index.pop(y * stride + x, None)
                game.cells_removed_this_step = True
        apply_stationary_health_cost(game, cell, y, x)
        return

    if not attack_intent:
        if target_value != 0:
            apply_stationary_health_cost(game, cell, y, x)
            return
        old_key = y * stride + x
        new_key = new_y * stride + new_x
        cell.y = new_y
        cell.x = new_x
        cell.pos = [new_y, new_x]
        grid[new_y, new_x] = 1
        index.pop(old_key, None)
        index[new_key] = cell
        grid[y, x] = 0
        cell.health -= MOVEMENT_HEALTH_COST
        if cell.health <= 0:
            grid[new_y, new_x] = 0
            index.pop(new_key, None)
            game.cells_removed_this_step = True
        apply_stationary_health_cost(game, cell, y, x)
        return

    if target_value == 0:
        apply_stationary_health_cost(game, cell, y, x)
        return

    ncell = index.get(new_y * stride + new_x, False)
    if ncell == False:
        if new_y < 2 or new_y > game.size.lines - 2 or new_x < 2 or new_x > game.size.columns - 2:
            grid[new_y, new_x] = -1
        else:
            grid[new_y, new_x] = 0
        apply_stationary_health_cost(game, cell, y, x)
        return

    ncell.health -= attack_damage_for_target(game, new_y, new_x, cell)
    success = ncell.health <= 0
    if success:
        grid[new_y, new_x] = 0
        index.pop(new_y * stride + new_x, None)
        game.cells_removed_this_step = True
        cell.add_health(KILL_HEALTH_REWARD)
        old_key = y * stride + x
        new_key = new_y * stride + new_x
        cell.y = new_y
        cell.x = new_x
        cell.pos = [new_y, new_x]
        grid[new_y, new_x] = 1
        index.pop(old_key, None)
        index[new_key] = cell
        grid[y, x] = 0
        game.add_cell(y, x, game.mutate(cell))
    else:
        cell.health -= 1
        if cell.health <= 0:
            grid[y, x] = 0
            index.pop(y * stride + x, None)
            game.cells_removed_this_step = True
    apply_stationary_health_cost(game, cell, y, x)


def step_sequential(game):
    cells = list(game.cells)
    game.defer_cell_list_removals = True
    game.cells_removed_this_step = False
    grid = game.grid
    index = game.cells_by_pos
    stride = game._cell_key_stride
    try:
        for cell in cells:
            if index.get(cell.y * stride + cell.x) is not cell:
                continue
            y, x = cell.y, cell.x
            neighbors = grid[y-2:y+3, x-2:x+3].reshape(25)
            action = cell.forward_flat_neighbors25(neighbors)
            apply_cell_action(game, cell, action)
    finally:
        game.defer_cell_list_removals = False
        if game.cells_removed_this_step:
            game.compact_cells()

    return game


def step(game):
    if getattr(game, 'action_backend', ACTION_BACKEND_SEQUENTIAL) != ACTION_BACKEND_FAMILY_BATCHED:
        return step_sequential(game)

    # Iterate over a snapshot because combat can remove cells before their turn.
    # The membership check preserves the original order for surviving cells while
    # preventing removed cells from acting later in the same frame.
    cells = list(game.cells)
    planned_actions = None
    if getattr(game, 'action_backend', ACTION_BACKEND_SEQUENTIAL) == ACTION_BACKEND_FAMILY_BATCHED:
        planned_actions = planned_family_action_list(game, cells)

    game.defer_cell_list_removals = True
    game.cells_removed_this_step = False
    grid = game.grid
    index = game.cells_by_pos
    stride = game._cell_key_stride
    try:
        for cell_order, cell in enumerate(cells):
            if index.get(cell.y * stride + cell.x) is not cell:
                continue
            y, x = cell.y, cell.x
            if planned_actions is None:
                # get the surrounding positions
                neighbors = grid[y-2:y+3, x-2:x+3].reshape(25)
                action = cell.forward_flat_neighbors25(neighbors)
            else:
                planned = planned_actions[cell_order]
                if planned is None:
                    neighbors = grid[y-2:y+3, x-2:x+3].reshape(25)
                    action = cell.forward_flat_neighbors25(neighbors)
                else:
                    action, hidden = planned
                    cell.commit_forward_state(hidden)

            apply_cell_action(game, cell, action)
    finally:
        game.defer_cell_list_removals = False
        if game.cells_removed_this_step:
            game.compact_cells()
            
    return game    


def init(game, num=2500):
    num = min(num, len(empty_positions(game)))
    factored_wave_family = game.make_factored_wave_family()
    weight_1, weight_2, coeff_1, coeff_2 = game.wave_factored_tensors(factored_wave_family, num)
    for index in range(num):
        new_cell = random_spawn(game, check_available=False)
        game.add_factored_cell(
            new_cell[0],
            new_cell[1],
            factored_wave_family,
            coeff_1[index],
            coeff_2[index],
            weight_1[index],
            weight_2[index],
        )

    return game

ROUNDTIME = 500
ROUND_TRANSITION_HEALTH_COST = 0
MOVEMENT_HEALTH_COST = 0.1
STATIONARY_HEALTH_COST = 1
STATIONARY_DAMAGE_AFTER_STEPS = 3
KILL_HEALTH_REWARD = 5
FITNESS_UPDATE_LR = 0.5
PER_WAVE = 600
MIN_WAVE = 500
MAX_TOTAL = 1000

def prune(game):    
    # prune the game if it is too big
    # damage all cells until it is small enough
    while len(game.cells) > PER_WAVE:
        game.damage_cell(random.choice(game.cells))
    return game


def main_tensor_rank1(args):
    global FITNESS_UPDATE_LR
    FITNESS_UPDATE_LR = args.fitness_update_lr

    if args.load:
        raise ValueError('--engine tensor_rank1 does not support --load yet')
    if args.save_on_complete:
        raise ValueError('--engine tensor_rank1 does not support --save-on-complete yet')
    from tensor_rank1_sim import (
        CudaGraphFamilyBasisBlockRunner,
        TensorRank1State,
        resolve_device,
        resolve_health_dtype,
        resolve_network_dtype,
        synchronize,
    )

    seed_all(args.seed)
    size = terminal_size(args.size)
    device_name = args.action_device
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = resolve_device(device_name)
    health_dtype = resolve_health_dtype(args.tensor_health_dtype)
    network_dtype = resolve_network_dtype(args.tensor_network_dtype, device)
    if args.tensor_matmul_precision is not None:
        torch.set_float32_matmul_precision(args.tensor_matmul_precision)
    board_capacity = (size.lines - 2) * size.columns
    initial_cells = min(args.initial_cells, board_capacity)

    state = TensorRank1State.fixed_capacity(
        active_cells=initial_cells,
        height=size.lines,
        width=size.columns,
        active_families=1,
        family_capacity=args.tensor_family_capacity,
        device=device,
        initial_health=args.tensor_initial_health,
        cell_capacity=args.tensor_cell_capacity,
        health_dtype=health_dtype,
        network_dtype=network_dtype,
        coeff_scale=args.tensor_coeff_scale,
        stationary_health_cap=args.tensor_stationary_health_cap,
    )
    active_family_count = 1
    rounds = 0
    countdown = ROUNDTIME
    frame = 0
    renders = 0
    snapshots = 0
    waves_spawned = 0
    empty_refills = 0
    graph_runner = None
    block_steps = max(int(args.tensor_block_steps), 1)
    if args.tensor_static_refill_check_every % block_steps != 0:
        raise ValueError('--tensor-block-steps must divide --tensor-static-refill-check-every')
    render_every = args.tensor_render_every if args.tensor_render_every is not None else block_steps
    if device.type == 'cuda':
        compile_state = state.clone()
        compile_state.compiled_snapshot_combat_steps(
            block_steps,
            rebuild_grid=True,
            family_basis=True,
            compile_mode=args.tensor_compile_mode,
        )
        synchronize(device)
    if device.type == 'cuda' and not args.no_tensor_cuda_graph:
        graph_runner = CudaGraphFamilyBasisBlockRunner(state, block_steps, args.tensor_compile_mode)

    last_active_cells = None

    def active_cell_count():
        nonlocal last_active_cells
        if last_active_cells is None:
            last_active_cells = int((state.health > 0).sum().item())
        return last_active_cells

    def invalidate_active_count():
        nonlocal last_active_cells
        last_active_cells = None

    def apply_spawn_count(spawned):
        nonlocal last_active_cells
        if last_active_cells is not None:
            last_active_cells += int(spawned)

    def spawn_wave(count):
        nonlocal active_family_count, graph_runner
        previous_family_version = state.family_capacity_version()
        spawned, active_family_count = state.append_static_weighted_wave(
            active_family_count,
            count,
            initial_health=args.tensor_wave_initial_health,
            coeff_scale=args.tensor_coeff_scale,
        )
        if state.family_capacity_version() != previous_family_version:
            graph_runner = None
        apply_spawn_count(spawned)
        return spawned

    def run_steps(step_count):
        nonlocal graph_runner
        if step_count <= 0:
            return
        if device.type == 'cuda':
            if not args.no_tensor_cuda_graph and step_count == block_steps:
                if graph_runner is None:
                    compile_state = state.clone()
                    compile_state.compiled_snapshot_combat_steps(
                        block_steps,
                        rebuild_grid=True,
                        family_basis=True,
                        compile_mode=args.tensor_compile_mode,
                    )
                    synchronize(device)
                    graph_runner = CudaGraphFamilyBasisBlockRunner(state, block_steps, args.tensor_compile_mode)
                graph_runner.replay()
            else:
                state.compiled_snapshot_combat_steps(
                    step_count,
                    rebuild_grid=True,
                    family_basis=True,
                    compile_mode=args.tensor_compile_mode,
                )
        else:
            for _ in range(step_count):
                state.step(movement='snapshot_combat', compact_dead=False, sync_positions=False)
        invalidate_active_count()

    def tensor_snapshot():
        synchronize(device)
        return (
            state.index_grid.detach().cpu().numpy(),
            state.health.detach().cpu().numpy(),
        )

    cursor_hidden = not args.no_render
    last_render_frame = None
    started = time.perf_counter()
    if cursor_hidden:
        hide_cursor()
    try:
        while args.max_frames is None or frame < args.max_frames:
            if frame % args.tensor_static_refill_check_every == 0 and active_cell_count() == 0:
                spawned = spawn_wave(MIN_WAVE)
                waves_spawned += spawned
                empty_refills += 1
                if spawned == 0:
                    break

            if args.snapshot_dir and frame % args.snapshot_every == 0:
                index_grid, health = tensor_snapshot()
                write_tensor_snapshot(index_grid, health, size, rounds, countdown, frame, args.snapshot_dir)
                snapshots += 1
            should_render = (
                not args.no_render
                and (last_render_frame is None or frame - last_render_frame >= render_every)
            )
            if should_render:
                index_grid, health = tensor_snapshot()
                draw_tensor_frame(index_grid, health, size, rounds, countdown, first_frame=(frame == 0))
                last_render_frame = frame
                renders += 1

            if args.frame_rate > 0:
                time.sleep(args.frame_rate)

            remaining = block_steps
            if args.max_frames is not None:
                remaining = min(remaining, args.max_frames - frame)
            step_count = min(remaining, countdown)
            run_steps(step_count)
            countdown -= step_count
            frame += step_count
            if countdown == 0:
                state.apply_round_transition_health_cost()
                invalidate_active_count()
                spawned = spawn_wave(max(PER_WAVE - active_cell_count(), MIN_WAVE))
                waves_spawned += spawned
                countdown = ROUNDTIME
                rounds += 1
    finally:
        if cursor_hidden:
            show_cursor()
    synchronize(device)
    elapsed = time.perf_counter() - started
    if args.metrics_json:
        metrics = {
            'active_cells_final': active_cell_count(),
            'action_device': str(device),
            'cell_capacity': state.cells,
            'cuda_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else '',
            'elapsed_seconds': elapsed,
            'empty_refills': empty_refills,
            'engine': SIM_ENGINE_TENSOR_RANK1,
            'family_capacity': state.families,
            'frames': frame,
            'frames_per_second': frame / elapsed if elapsed > 0 else 0.0,
            'render_every': render_every,
            'renders': renders,
            'rounds': rounds,
            'snapshots': snapshots,
            'tensor_coeff_scale': args.tensor_coeff_scale,
            'fitness_update_lr': FITNESS_UPDATE_LR,
            'tensor_network_dtype': str(network_dtype).removeprefix('torch.'),
            'tensor_network_dtype_requested': args.tensor_network_dtype,
            'tensor_stationary_health_cap': args.tensor_stationary_health_cap,
            'tensor_block_steps': block_steps,
            'waves_spawned': waves_spawned,
        }
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def main(args):
    if args.engine == SIM_ENGINE_TENSOR_RANK1:
        main_tensor_rank1(args)
        return

    seed_all(args.seed)
    with torch.inference_mode():
        if args.load:
            game = load_state(args.load)
            game.action_backend = args.action_backend
            game.action_device = args.action_device
            game.batched_min_family_size = args.batched_min_family_size
        else:
            game = init(Game(
                size=args.size,
                mutation_mode=args.mutation_mode,
                action_backend=args.action_backend,
                action_device=args.action_device,
                batched_min_family_size=args.batched_min_family_size,
            ), num=args.initial_cells)

        countdown = ROUNDTIME
        frame = 0
        cursor_hidden = not args.no_render
        if cursor_hidden:
            hide_cursor()
        try:
            while args.max_frames is None or frame < args.max_frames:
                try:
                    if len(game.cells) == 0:
                        game = init(game, num=MIN_WAVE)
                    # Snapshot flags are for bounded tests/reviews only. They run
                    # before step(), matching what a user sees rendered each frame.
                    if args.snapshot_dir and frame % args.snapshot_every == 0:
                        write_snapshot(game, countdown, frame, args.snapshot_dir)
                    if not args.no_render:
                        draw_frame(game, countdown, first_frame=(frame == 0))
                    game, countdown = advance_round(game, countdown)

                    if args.frame_rate > 0:
                        time.sleep(args.frame_rate)
                    game = step(game)
                    countdown -= 1
                    frame += 1

                except KeyboardInterrupt:
                    if cursor_hidden:
                        show_cursor()
                    print('Saving...')
                    save_state(game, args.save)
                    print('Saved!')
                    break
        finally:
            if cursor_hidden:
                show_cursor()
        if args.save_on_complete:
            save_state(game, args.save)

def save_state(game, name):
    with open(name, 'wb') as f:
        pickle.dump(game, f)

def load_state(name):
    with open(name, 'rb') as f:
        game = pickle.load(f)
    if not hasattr(game, '_cell_key_stride'):
        game._cell_key_stride = game.grid.shape[1]
    if not hasattr(game, 'defer_cell_list_removals'):
        game.defer_cell_list_removals = False
    if not hasattr(game, 'cells_removed_this_step'):
        game.cells_removed_this_step = False
    for cell in game.cells:
        if not hasattr(cell, 'y') or not hasattr(cell, 'x'):
            cell.y = int(cell.pos[0])
            cell.x = int(cell.pos[1])
        cell.pos = [int(cell.y), int(cell.x)]
        if getattr(cell, 'prev_state', None) is None or cell.prev_state.shape[0] != HIDDEN_DIM:
            cell.prev_state = np.zeros(HIDDEN_DIM, dtype=np.float32)
    game._rebuild_cell_index()
    return game




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Petri Dish')
    parser.add_argument('--engine', choices=SIM_ENGINES, default=SIM_ENGINE_GAME)
    parser.add_argument('--load', help='load a saved state', default=False)
    parser.add_argument('--save', help='save the state', default='state.pkl')
    parser.add_argument('--save-on-complete', action='store_true', help='save when a bounded run finishes')
    parser.add_argument('--seed', type=int, help='seed Python, NumPy, and PyTorch RNGs')
    parser.add_argument('--size', type=parse_size, help='grid size as LINESxCOLUMNS, for example 24x80')
    parser.add_argument('--initial-cells', type=positive_int, default=2500, help='number of cells to spawn initially')
    parser.add_argument('--mutation-mode', choices=MUTATION_MODES, default=DEFAULT_MUTATION_MODE)
    parser.add_argument(
        '--action-backend',
        choices=ACTION_BACKENDS,
        default=ACTION_BACKEND_SEQUENTIAL,
        help='action evaluator; family_batched is experimental snapshot semantics, sequential is exact',
    )
    parser.add_argument('--action-device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument(
        '--batched-min-family-size',
        type=positive_int,
        default=32,
        help='minimum compatible family size for the batched action backend',
    )
    parser.add_argument('--max-frames', type=positive_int, help='stop after this many frames')
    parser.add_argument('--snapshot-dir', help='write plain-text frame snapshots to this directory')
    parser.add_argument('--snapshot-every', type=positive_int, default=1, help='write one snapshot every N frames')
    parser.add_argument('--metrics-json', help='write bounded-run metrics JSON for benchmark/debug runs')
    parser.add_argument('--no-render', action='store_true', help='do not render frames to the terminal')
    parser.add_argument('--frame-rate', type=float, default=FRAME_RATE, help='seconds to sleep between frames')
    parser.add_argument('--tensor-block-steps', type=positive_int, default=10)
    parser.add_argument('--tensor-render-every', type=positive_int)
    parser.add_argument('--tensor-family-capacity', type=positive_int, default=10)
    parser.add_argument('--tensor-cell-capacity', type=positive_int)
    parser.add_argument('--tensor-initial-health', type=positive_int, default=15)
    parser.add_argument('--tensor-wave-initial-health', type=positive_int, default=2)
    parser.add_argument('--tensor-coeff-scale', type=float, default=FACTORED_WAVE_COEFF_SCALE)
    parser.add_argument('--fitness-update-lr', type=float, default=FITNESS_UPDATE_LR)
    parser.add_argument('--tensor-stationary-health-cap', type=int, default=1)
    parser.add_argument('--tensor-static-refill-check-every', type=positive_int, default=100)
    parser.add_argument('--tensor-health-dtype', choices=('float32', 'int64', 'int32'), default='float32')
    parser.add_argument('--tensor-network-dtype', choices=('auto', 'float32', 'float16', 'bfloat16'), default='auto')
    parser.add_argument('--tensor-compile-mode', choices=('default', 'reduce-overhead', 'max-autotune'), default='default')
    parser.add_argument('--tensor-matmul-precision', choices=('highest', 'high', 'medium'), default='high')
    parser.add_argument('--tensor-cuda-graph', dest='no_tensor_cuda_graph', action='store_false')
    parser.add_argument('--no-tensor-cuda-graph', dest='no_tensor_cuda_graph', action='store_true')
    parser.set_defaults(no_tensor_cuda_graph=True)
    args = parser.parse_args()
    if (
            args.engine == SIM_ENGINE_TENSOR_RANK1
            and args.tensor_static_refill_check_every % args.tensor_block_steps != 0):
        parser.error('--tensor-block-steps must divide --tensor-static-refill-check-every')

    main(args)
