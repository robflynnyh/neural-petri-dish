import numpy as np
import torch
import torch.nn as nn
import argparse
import os
from pathlib import Path
import pickle
import random
import shutil
import sys
import time

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


def clone_parameter(tensor):
    return nn.Parameter(tensor.detach().clone())


def clone_genes(genes):
    return {key: value.detach().clone() for key, value in genes.items()}


def empty_positions(game):
    play_area = game.grid[2:game.size.lines, 2:game.size.columns + 2]
    positions = np.argwhere(play_area == 0)
    return positions + np.array([2, 2])


col = 16
X = f'{bg(col)}❏{bg.rs}'# the icon for the cell *·◉ ○ ●○○✺✺
BLANK = f'{bg(col)} {bg.rs}' # icon for empty cell °
FRAME_RATE = 0.05    # seconds between frames
MUTATION_MODE_LEGACY = 'legacy'
MUTATION_MODE_LOW_RANK = 'low_rank'
MUTATION_MODE_SHARED_RANK1 = 'shared_rank1'
MUTATION_MODES = (MUTATION_MODE_LEGACY, MUTATION_MODE_LOW_RANK, MUTATION_MODE_SHARED_RANK1)
DEFAULT_MUTATION_MODE = MUTATION_MODE_LOW_RANK
LOW_RANK_MUTATION_RANK = 2
SHARED_RANK1_PERTURBATION_SCALE = 0.03
ACTION_MODE_SEQUENTIAL = 'sequential'
ACTION_MODE_SIMULTANEOUS = 'simultaneous'
ACTION_MODES = (ACTION_MODE_SEQUENTIAL, ACTION_MODE_SIMULTANEOUS)
DEFAULT_ACTION_MODE = ACTION_MODE_SEQUENTIAL

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

class Cell(nn.Module):
    '''
    Manages each cell, it's genes and it's neural network
    '''
    def __init__(self, pos, genes=None):
        super(Cell, self).__init__()
        self.linear = nn.Linear(33, 9)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(9, 9)
        self.pos = list(pos)
        self.health = 2
        self.max_health = 15
        self.age = 0
        self.prev_state = None
        self.diversity = None

        self.pos_list = []
        self.genome_mode = 'full'
        self.base_id = None
        self.shared_base = None
        self.rank1 = None

        if genes is not None:
            self.genome_mode = genes.get('genome_mode', 'full')
            self.base_id = genes.get('base_id')
            if self.genome_mode == MUTATION_MODE_SHARED_RANK1:
                self.shared_base = clone_genes(genes['shared_base'])
                self.rank1 = clone_genes(genes['rank1'])
            self.linear.weight = clone_parameter(genes['weight_1'])
            self.linear.bias = clone_parameter(genes['bias_1'])
            self.linear2.weight = clone_parameter(genes['weight_2'])
            self.linear2.bias = clone_parameter(genes['bias_2'])

    def update_position_history(self):
        self.pos_list.append(self.pos)
        if len(self.pos_list) > 1:
            same_count = 0
            for el in self.pos_list[-5:-2]:
                if el == self.pos:
                    same_count += 1
            if same_count > 1: # if the cell has been in the same location for 3 frames, its health goes to 1 # NOW 1
                self.health = 1
                self.pos_list = []

    def forward(self, neighbors):
        inps = neighbors.unsqueeze(0)
        if self.prev_state is not None:
            all_imps = torch.cat((inps, self.prev_state), dim=-1)
        else:
            state = torch.zeros(1, 1, 9, dtype=inps.dtype, device=inps.device)
            all_imps = torch.cat((inps, state), dim=-1)
        lin1 = self.relu(self.linear(all_imps))
        self.prev_state = lin1.detach()
        lin2 = self.linear2(lin1)

        self.update_position_history()
        return lin2.argmax()

    def update_pos(self, pos):
        self.pos = pos.tolist() if isinstance(pos, np.ndarray) else list(pos)
                    
            

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
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def random_spawn(game):
    '''returns a random position in the grid that is not occupied'''
    if len(empty_positions(game)) == 0:
        raise RuntimeError('No Empty Positions Available')
    # Keep the original rejection-sampling draw order so seeded runs match the
    # interactive simulation as closely as possible. The pre-check only prevents
    # an infinite loop when the grid is full.
    while True:
        pos = np.array([np.random.randint(2, game.size.lines), np.random.randint(2, game.size.columns+2)])
        if game.grid[pos[0], pos[1]] == 0:
            return pos


def structured_noise_like(tensor, scale, rank=LOW_RANK_MUTATION_RANK):
    if tensor.ndim != 2:
        return torch.randn_like(tensor) * scale

    rank = min(rank, *tensor.shape)
    left = torch.randn_like(tensor[:, :rank])
    right = torch.randn_like(tensor[:rank, :])
    # EGGROLL-style low-rank noise perturbs a weight matrix through a factored
    # direction instead of independent noise at every parameter. The sqrt(rank)
    # normalization keeps the entry scale close to ordinary Gaussian noise.
    return left.matmul(right) * (scale / rank ** 0.5)


def mutation_noise_like(tensor, scale, mode=DEFAULT_MUTATION_MODE):
    if mode == MUTATION_MODE_LEGACY:
        return torch.randn_like(tensor) * scale
    if mode == MUTATION_MODE_LOW_RANK:
        return structured_noise_like(tensor, scale)
    raise ValueError(f'Unknown mutation mode: {mode}')


def add_mutation_noise(tensor, scale, mode=DEFAULT_MUTATION_MODE):
    return tensor + mutation_noise_like(tensor, scale, mode)


def rank1_delta(u, v):
    return u.unsqueeze(1).matmul(v.unsqueeze(0))


def random_rank1_factors(base_genes, scale=SHARED_RANK1_PERTURBATION_SCALE):
    return {
        'weight_1_u': torch.randn(base_genes['weight_1'].shape[0]) * scale,
        'weight_1_v': torch.randn(base_genes['weight_1'].shape[1]) / base_genes['weight_1'].shape[1] ** 0.5,
        'bias_1_delta': torch.randn_like(base_genes['bias_1']) * scale,
        'weight_2_u': torch.randn(base_genes['weight_2'].shape[0]) * scale,
        'weight_2_v': torch.randn(base_genes['weight_2'].shape[1]) / base_genes['weight_2'].shape[1] ** 0.5,
        'bias_2_delta': torch.randn_like(base_genes['bias_2']) * scale,
    }


def materialize_shared_rank1_genes(base_genes, rank1_factors):
    return {
        'weight_1': base_genes['weight_1'] + rank1_delta(rank1_factors['weight_1_u'], rank1_factors['weight_1_v']),
        'bias_1': base_genes['bias_1'] + rank1_factors['bias_1_delta'],
        'weight_2': base_genes['weight_2'] + rank1_delta(rank1_factors['weight_2_u'], rank1_factors['weight_2_v']),
        'bias_2': base_genes['bias_2'] + rank1_factors['bias_2_delta'],
    }


def make_shared_rank1_genes(base_genes, base_id, scale=SHARED_RANK1_PERTURBATION_SCALE):
    rank1_factors = random_rank1_factors(base_genes, scale)
    genes = materialize_shared_rank1_genes(base_genes, rank1_factors)
    genes.update({
        'genome_mode': MUTATION_MODE_SHARED_RANK1,
        'base_id': base_id,
        'shared_base': clone_genes(base_genes),
        'rank1': clone_genes(rank1_factors),
    })
    return genes


class Game():
    '''
    Manages Game State
    '''
    def __init__(self, genepool=None, size=None, mutation_mode=DEFAULT_MUTATION_MODE, action_mode=DEFAULT_ACTION_MODE):
        self.size = terminal_size(size)
        self.grid = np.zeros((self.size.lines+2, self.size.columns+4))
        # set the 2 layer border as -1's (each cell has a vision of 4x4 hence border to avoid out of bounds)
        self.grid[:, 0:2] = -1
        self.grid[:, -2:] = -1
        self.grid[0:2, :] = -1
        self.grid[-2:, :] = -1
        self.rounds = 0
        #
        self.cells = []
        #self.graveyard = []
        self.mutate_rate = 0.00001
        self.mutation_mode = mutation_mode
        self.action_mode = action_mode
        self.shared_base_genes = None
        self.shared_base_id = 0

    def ensure_shared_base(self):
        if self.shared_base_genes is None:
            self.shared_base_genes = clone_genes(Cell([2, 2]).get_genes())
        return self.shared_base_genes

    def shared_rank1_genes(self, base_genes=None, base_id=None):
        if base_genes is None:
            base_genes = self.ensure_shared_base()
        if base_id is None:
            base_id = self.shared_base_id
        return make_shared_rank1_genes(base_genes, base_id)

    def update_shared_base_from_survivors(self):
        if self.mutation_mode != MUTATION_MODE_SHARED_RANK1 or not self.cells:
            return

        total_health = sum(max(cell.health, 0) for cell in self.cells)
        if total_health <= 0:
            return

        averaged = {}
        for key in ('weight_1', 'bias_1', 'weight_2', 'bias_2'):
            value = None
            for cell in self.cells:
                weight = max(cell.health, 0) / total_health
                contribution = cell.get_genes()[key].detach() * weight
                value = contribution if value is None else value + contribution
            averaged[key] = value

        self.shared_base_genes = averaged
        self.shared_base_id += 1

    def get_cell(self, y, x):
        cell = [c for c in self.cells if c.pos == [y, x]]
        if len(cell) != 0:
            return cell[0]
        else:
            return False

    def update_cell(self, y, x, new, cell=None):
        new = [int(new[0]), int(new[1])]
        if cell is None:
            cell = self.get_cell(y, x)
            if cell == False:
                raise Exception('Cell Does not Exist at this Position')
        occupant = self.get_cell(*new)
        if occupant is not False and occupant is not cell:
            raise Exception('New Position is Occupied')
        if self.grid[new[0], new[1]] == -1:
            raise Exception('New Position is Outside the Play Area')
       
        cell.update_pos(new)
        self.grid[new[0], new[1]] = 1
        if [y, x] != new:
            self.grid[y][x] = 0

    def remove_cell(self, y, x): 
        #self.graveyard.append(self.get_cell(y, x)) # change so cell is passed in
        self.grid[y, x] = 0
        self.cells = [c for c in self.cells if c.pos != [y, x]]

    def apply_damage(self, cell, amount=1):
        if cell == False or cell not in self.cells:
            return False
        cell.health -= amount
        return cell.health <= 0

    def kill_cell(self, cell):
        if cell == False or cell not in self.cells:
            return False
        self.remove_cell(*cell.pos)
        return True

    def damage_cell(self, cell, amount=1):
        if self.apply_damage(cell, amount=amount):
            self.remove_cell(*cell.pos)
            return True
        return False

    def add_cell(self, y, x, genes=None):
        if self.grid[y][x] != 0:
            raise Exception('Cannot Add Cell to a Non-Empty Position')
        self.grid[y][x] = 1
        self.cells.append(Cell([y, x], genes))

    def mutate(self, cell):
        '''
        Mutates a cell's genes bashed on the surrounding cells
        '''
        mutation_mode = getattr(self, 'mutation_mode', DEFAULT_MUTATION_MODE)
        if mutation_mode == MUTATION_MODE_SHARED_RANK1:
            base_genes = cell.shared_base if cell.shared_base is not None else self.ensure_shared_base()
            base_id = cell.base_id if cell.base_id is not None else self.shared_base_id
            return self.shared_rank1_genes(base_genes, base_id)

        # 5% chance of larger mutation. In low_rank mode, matrix weights receive
        # an EGGROLL-inspired factored perturbation; legacy mode preserves the
        # old independent Gaussian noise for baseline comparisons.
        
        if np.random.rand() < 0.05:
            # get the surrounding positions
            neighbors = [direction_dict[i](cell.pos) for i in range(1,9)]
            ncells = [self.get_cell(*n) for n in neighbors]
            ncells = [c for c in ncells if c != False]
            if len(ncells) != 0:
                genes = [c.get_genes() for c in ncells]
                # combine the genes 
                new_genes = {}
                for k in genes[0].keys():
                    new_genes[k] = torch.mean(torch.stack([g[k] for g in genes]), dim=0)
                # Blend local neighbor genes, then add a low-rank matrix
                # perturbation so inherited policies shift coherently rather
                # than by independent noise at every weight.
                weight1 = add_mutation_noise(cell.linear.weight * 0.8 + new_genes['weight_1'] * 0.2, 0.1, mutation_mode)
                bias1 = add_mutation_noise(cell.linear.bias * 0.8 + new_genes['bias_1'] * 0.2, 0.1, mutation_mode)
                weight2 = add_mutation_noise(cell.linear2.weight * 0.8 + new_genes['weight_2'] * 0.2, 0.1, mutation_mode)
                bias2 = add_mutation_noise(cell.linear2.bias * 0.8 + new_genes['bias_2'] * 0.2, 0.1, mutation_mode)
            else:
                weight1 = add_mutation_noise(cell.linear.weight, 0.001, mutation_mode)
                bias1 = add_mutation_noise(cell.linear.bias, 0.001, mutation_mode)
                weight2 = add_mutation_noise(cell.linear2.weight, 0.001, mutation_mode)
                bias2 = add_mutation_noise(cell.linear2.bias, 0.001, mutation_mode)
        # This intentionally remains a second RNG draw, matching the original
        # probability path. Collapsing it into one draw changes seeded dynamics.
        elif np.random.rand() < 0.4:
            weight1 = add_mutation_noise(cell.linear.weight, 0.00001, mutation_mode)
            bias1 = add_mutation_noise(cell.linear.bias, 0.00001, mutation_mode)
            weight2 = add_mutation_noise(cell.linear2.weight, 0.00001, mutation_mode)
            bias2 = add_mutation_noise(cell.linear2.bias, 0.00001, mutation_mode)
        else:
            # no mutation
            weight1 = cell.linear.weight.detach().clone()
            bias1 = cell.linear.bias.detach().clone()
            weight2 = cell.linear2.weight.detach().clone()
            bias2 = cell.linear2.bias.detach().clone()
        return {'weight_1': weight1, 'bias_1': bias1,
                'weight_2': weight2, 'bias_2': bias2}


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


def advance_round(game, countdown):
    if countdown == 0:
        totalcells = len(game.cells)
        maxage = max([c.age for c in game.cells]) if game.cells else 0
        game.update_shared_base_from_survivors()
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


def neighbor_tensor_for_cell(game, cell):
    y, x = cell.pos
    neighbors = np.delete(game.grid[y-2:y+3, x-2:x+3].reshape(-1), 12, 0)
    if neighbors.shape[0] != 24:
        raise RuntimeError(f'Expected 24 Neighbor Inputs, Got {neighbors.shape[0]}')
    return torch.from_numpy(neighbors.astype(np.float32, copy=False))


def batched_shared_rank1_actions(cells, neighbor_tensors):
    inputs = []
    for cell, neighbors in zip(cells, neighbor_tensors):
        if cell.prev_state is None:
            state = torch.zeros(9, dtype=neighbors.dtype, device=neighbors.device)
        else:
            state = cell.prev_state.reshape(-1).to(dtype=neighbors.dtype, device=neighbors.device)
        inputs.append(torch.cat((neighbors, state), dim=-1))
    x = torch.stack(inputs)

    base = cells[0].shared_base
    w1 = base['weight_1'].to(dtype=x.dtype, device=x.device)
    b1 = base['bias_1'].to(dtype=x.dtype, device=x.device)
    w2 = base['weight_2'].to(dtype=x.dtype, device=x.device)
    b2 = base['bias_2'].to(dtype=x.dtype, device=x.device)

    u1 = torch.stack([cell.rank1['weight_1_u'].to(dtype=x.dtype, device=x.device) for cell in cells])
    v1 = torch.stack([cell.rank1['weight_1_v'].to(dtype=x.dtype, device=x.device) for cell in cells])
    b1_delta = torch.stack([cell.rank1['bias_1_delta'].to(dtype=x.dtype, device=x.device) for cell in cells])
    hidden = x.matmul(w1.t()) + b1
    hidden = hidden + (x * v1).sum(dim=1, keepdim=True) * u1 + b1_delta
    hidden = torch.relu(hidden)

    u2 = torch.stack([cell.rank1['weight_2_u'].to(dtype=x.dtype, device=x.device) for cell in cells])
    v2 = torch.stack([cell.rank1['weight_2_v'].to(dtype=x.dtype, device=x.device) for cell in cells])
    b2_delta = torch.stack([cell.rank1['bias_2_delta'].to(dtype=x.dtype, device=x.device) for cell in cells])
    logits = hidden.matmul(w2.t()) + b2
    logits = logits + (hidden * v2).sum(dim=1, keepdim=True) * u2 + b2_delta

    for cell, state in zip(cells, hidden.detach()):
        cell.prev_state = state.reshape(1, 1, -1)
        cell.update_position_history()
    return logits.argmax(dim=1).int().tolist()


def compute_actions(game, cells):
    actions = {}
    grouped = {}
    individual = []
    for cell in cells:
        if cell not in game.cells:
            continue
        if cell.genome_mode == MUTATION_MODE_SHARED_RANK1 and cell.shared_base is not None and cell.rank1 is not None:
            grouped.setdefault(cell.base_id, []).append(cell)
        else:
            individual.append(cell)

    for group in grouped.values():
        neighbors = [neighbor_tensor_for_cell(game, cell) for cell in group]
        for cell, action in zip(group, batched_shared_rank1_actions(group, neighbors)):
            actions[id(cell)] = action

    for cell in individual:
        neighbor_tensor = neighbor_tensor_for_cell(game, cell).unsqueeze(0)
        actions[id(cell)] = cell(neighbor_tensor).int().tolist()
    return actions


def move_cell_if_open(game, cell, target):
    if cell not in game.cells:
        return False
    if game.grid[target[0], target[1]] != 0:
        return False
    game.update_cell(*cell.pos, target, cell)
    return True


def resolve_successful_attack(game, attacker, victim):
    if attacker not in game.cells or victim not in game.cells:
        return False
    attacker_origin = list(attacker.pos)
    victim_pos = list(victim.pos)
    game.kill_cell(victim)
    attacker.add_health()
    if game.grid[victim_pos[0], victim_pos[1]] == 0:
        game.update_cell(*attacker_origin, victim_pos, attacker)
        if game.grid[attacker_origin[0], attacker_origin[1]] == 0:
            game.add_cell(*attacker_origin, game.mutate(attacker))
    return True


def step_sequential(game):
    # Iterate over a snapshot because combat can remove cells before their turn.
    # The membership check preserves the original order for surviving cells while
    # preventing removed cells from acting later in the same frame.
    for cell in list(game.cells):
        if cell not in game.cells:
            continue
        y, x = cell.pos
        action = compute_actions(game, [cell])[id(cell)]

        if action != 0:
            new_loc = direction_dict[action]([y, x])
        

            if game.grid[new_loc[0], new_loc[1]] != -1:

                if game.grid[new_loc[0], new_loc[1]] == 0: # if the new location is empty
                    game.update_cell(y, x, new_loc, cell)
                else:
                    #sucess = game.damage_cell(game.get_cell(*new_loc)) # bites the other cell
                    ###
                    ncell = game.get_cell(*new_loc)
                    if ncell == False:
                        pos = [*new_loc]
                        if pos[0] < 2 or pos[0] > game.size.lines-2 or pos[1] < 2 or pos[1] > game.size.columns-2:
                            game.grid[pos[0], pos[1]] = -1
                        else:
                            game.grid[pos[0], pos[1]] = 0
                        continue
                    ###
                    if game.apply_damage(ncell):
                        resolve_successful_attack(game, cell, ncell)
                    else:
                        game.damage_cell(cell)
            else:
                game.damage_cell(cell)
        else:
            pass
            #game.damage_cell(cell) # if it doesn't move it loses health
            #game.remove_cell(y, x) # if it doesn't move it dies
            
    return game    


def step_simultaneous(game):
    acting_cells = list(game.cells)
    actions = compute_actions(game, acting_cells)
    start_positions = {tuple(cell.pos): cell for cell in acting_cells if cell in game.cells}
    start_grid = game.grid.copy()
    proposals = {}
    target_to_movers = {}

    for cell in acting_cells:
        if cell not in game.cells:
            continue
        action = actions.get(id(cell), 0)
        target = tuple(direction_dict[action](cell.pos).tolist())
        proposals[cell] = {'action': action, 'origin': tuple(cell.pos), 'target': target}
        if action != 0:
            target_to_movers.setdefault(target, []).append(cell)

    resolved = set()
    for cell, proposal in proposals.items():
        target = proposal['target']
        if proposal['action'] != 0 and start_grid[target[0], target[1]] == -1:
            game.damage_cell(cell)
            resolved.add(cell)

    for target, attackers in target_to_movers.items():
        if start_grid[target[0], target[1]] != 1:
            continue
        victim = start_positions.get(target)
        live_attackers = [cell for cell in attackers if cell in game.cells and cell is not victim]
        if victim is None or victim not in game.cells or not live_attackers:
            continue

        winner = random.choice(live_attackers)
        victim_proposal = proposals.get(victim)
        victim_moving = victim_proposal is not None and victim_proposal['action'] != 0 and victim_proposal['target'] != target
        if victim_moving:
            # A cell that tries to leave an attacked square gets an escape roll.
            # This removes deterministic first-mover advantage while keeping
            # combat local and cheap to resolve after batched action inference.
            caught = random.random() < 0.5
            if caught:
                resolve_successful_attack(game, winner, victim)
                resolved.add(winner)
                resolved.add(victim)
            for attacker in live_attackers:
                if attacker is not winner and attacker in game.cells:
                    game.damage_cell(attacker)
                    resolved.add(attacker)
            if not caught:
                for attacker in live_attackers:
                    if attacker in game.cells:
                        game.damage_cell(attacker)
                        resolved.add(attacker)
            continue

        if game.apply_damage(victim):
            resolve_successful_attack(game, winner, victim)
        else:
            game.damage_cell(winner)
        resolved.add(winner)
        resolved.add(victim)
        for attacker in live_attackers:
            if attacker is not winner and attacker in game.cells:
                game.damage_cell(attacker)
                resolved.add(attacker)

    for target, movers in target_to_movers.items():
        if start_grid[target[0], target[1]] != 0:
            continue
        candidates = [cell for cell in movers if cell in game.cells and cell not in resolved]
        if not candidates:
            continue
        winner = random.choice(candidates)
        move_cell_if_open(game, winner, list(target))
        resolved.add(winner)
        for loser in candidates:
            if loser is not winner and loser in game.cells:
                game.damage_cell(loser)
                resolved.add(loser)
    return game


def step(game):
    if getattr(game, 'action_mode', DEFAULT_ACTION_MODE) == ACTION_MODE_SIMULTANEOUS:
        return step_simultaneous(game)
    return step_sequential(game)


def init(game, num=2500):
    total_in_game = len(game.cells)
    num = min(num, len(empty_positions(game)))
    if game.mutation_mode == MUTATION_MODE_SHARED_RANK1:
        game.ensure_shared_base()
    for i in range(num):
        new_cell = random_spawn(game)
        if game.mutation_mode == MUTATION_MODE_SHARED_RANK1:
            # New wave cells share the current round base genome and differ
            # only by rank-1 perturbations. Survivors from older rounds retain
            # their own base_id, so extinct bases naturally disappear.
            game.add_cell(*new_cell, genes=game.shared_rank1_genes())
        # Preserve the original startup behavior: when the dish begins empty,
        # every initial cell is independently random. Only later wave spawns
        # mutate from the cells that existed before the wave started.
        elif total_in_game == 0:
            game.add_cell(*new_cell)
        else:
            cell = random.choice(game.cells)
            game.add_cell(*new_cell, genes=game.mutate(cell))
            total_in_game -= 1

    return game

ROUNDTIME = 500
PER_WAVE = 300
MIN_WAVE = 250
MAX_TOTAL = 1000

def prune(game):    
    # prune the game if it is too big
    # damage all cells until it is small enough
    while len(game.cells) > PER_WAVE:
        game.damage_cell(random.choice(game.cells))
    return game


def main(args):
    seed_all(args.seed)
    with torch.no_grad():
        if args.load:
            game = load_state(args.load)
        else:
            game = init(Game(size=args.size, mutation_mode=args.mutation_mode, action_mode=args.action_mode), num=args.initial_cells)

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
        return pickle.load(f)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Petri Dish')
    parser.add_argument('--load', help='load a saved state', default=False)
    parser.add_argument('--save', help='save the state', default='state.pkl')
    parser.add_argument('--save-on-complete', action='store_true', help='save when a bounded run finishes')
    parser.add_argument('--seed', type=int, help='seed Python, NumPy, and PyTorch RNGs')
    parser.add_argument('--size', type=parse_size, help='grid size as LINESxCOLUMNS, for example 24x80')
    parser.add_argument('--initial-cells', type=positive_int, default=2500, help='number of cells to spawn initially')
    parser.add_argument('--mutation-mode', choices=MUTATION_MODES, default=DEFAULT_MUTATION_MODE)
    parser.add_argument('--action-mode', choices=ACTION_MODES, default=DEFAULT_ACTION_MODE)
    parser.add_argument('--max-frames', type=positive_int, help='stop after this many frames')
    parser.add_argument('--snapshot-dir', help='write plain-text frame snapshots to this directory')
    parser.add_argument('--snapshot-every', type=positive_int, default=1, help='write one snapshot every N frames')
    parser.add_argument('--no-render', action='store_true', help='do not render frames to the terminal')
    parser.add_argument('--frame-rate', type=float, default=FRAME_RATE, help='seconds to sleep between frames')
    args = parser.parse_args()

    main(args)
