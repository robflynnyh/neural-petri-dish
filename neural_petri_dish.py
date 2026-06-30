import numpy as np
import torch
import torch.nn as nn
import argparse
import os
from pathlib import Path
import pickle
import random
import shutil
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
        os.system('clear')


def terminal_size(size=None):
    if size is None:
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


def empty_positions(game):
    play_area = game.grid[2:game.size.lines, 2:game.size.columns + 2]
    positions = np.argwhere(play_area == 0)
    return positions + np.array([2, 2])


col = 16
X = f'{bg(col)}❏{bg.rs}'# the icon for the cell *·◉ ○ ●○○✺✺
BLANK = f'{bg(col)} {bg.rs}' # icon for empty cell °
FRAME_RATE = 0.05    # seconds between frames

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

        if genes is not None:
            self.linear.weight = clone_parameter(genes['weight_1'])
            self.linear.bias = clone_parameter(genes['bias_1'])
            self.linear2.weight = clone_parameter(genes['weight_2'])
            self.linear2.bias = clone_parameter(genes['bias_2'])

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

        self.pos_list.append(self.pos)
        if len(self.pos_list) > 1:
            same_count = 0
            for el in self.pos_list[-5:-2]:
                if el == self.pos:
                    same_count += 1
            if same_count > 1: # if the cell has been in the same location for 3 frames, its health goes to 1 # NOW 1
                self.health = 1
                self.pos_list = []
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
    positions = empty_positions(game)
    if len(positions) == 0:
        raise RuntimeError('No Empty Positions Available')
    return positions[np.random.randint(len(positions))]

class Game():
    '''
    Manages Game State
    '''
    def __init__(self, genepool=None, size=None):
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

    def damage_cell(self, cell):
        if cell == False or cell not in self.cells:
            return False
        cell.health -= 1
        if cell.health <= 0:
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
        # 10% chance of mutation
        
        roll = np.random.rand()
        if roll < 0.05:
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
                # alter the genes of the cell through linear combination and add random noise with a small guassian
                weight1 = cell.linear.weight * 0.8 + new_genes['weight_1'] * 0.2 + torch.randn_like(cell.linear.weight) * 0.1
                bias1 = cell.linear.bias * 0.8 + new_genes['bias_1'] * 0.2 + torch.randn_like(cell.linear.bias) * 0.1
                weight2 = cell.linear2.weight * 0.8 + new_genes['weight_2'] * 0.2 + torch.randn_like(cell.linear2.weight) * 0.1
                bias2 = cell.linear2.bias * 0.8 + new_genes['bias_2'] * 0.2 + torch.randn_like(cell.linear2.bias) * 0.1
            else:
                weight1 = cell.linear.weight + torch.randn_like(cell.linear.weight) * 0.001
                bias1 = cell.linear.bias + torch.randn_like(cell.linear.bias) * 0.001
                weight2 = cell.linear2.weight + torch.randn_like(cell.linear2.weight) * 0.001
                bias2 = cell.linear2.bias + torch.randn_like(cell.linear2.bias) * 0.001
        elif roll < 0.45:
            weight1 = cell.linear.weight + torch.randn_like(cell.linear.weight) * 0.00001
            bias1 = cell.linear.bias + torch.randn_like(cell.linear.bias) * 0.00001
            weight2 = cell.linear2.weight + torch.randn_like(cell.linear2.weight) * 0.00001
            bias2 = cell.linear2.bias + torch.randn_like(cell.linear2.bias) * 0.00001
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


def step(game):
    for cell in list(game.cells):
        if cell not in game.cells:
            continue
        y, x = cell.pos
        # get the surrounding positions
        neighbors = np.delete(game.grid[y-2:y+3, x-2:x+3].reshape(-1), 12, 0) 
        if neighbors.shape[0] != 24:
            raise RuntimeError(f'Expected 24 Neighbor Inputs, Got {neighbors.shape[0]}')
        neighbor_tensor = torch.from_numpy(neighbors.astype(np.float32, copy=False)).unsqueeze(0)
        action = cell(neighbor_tensor).int().tolist()

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
                    sucess = game.damage_cell(ncell)
                    if sucess: # if the other cell dies 
                        cell.add_health() # gains health cus it ate !
                        game.update_cell(y, x, new_loc, cell)
                        game.add_cell(y, x, game.mutate(cell))
                    else:
                        game.damage_cell(cell)
            else:
                game.damage_cell(cell)
        else:
            pass
            #game.damage_cell(cell) # if it doesn't move it loses health
            #game.remove_cell(y, x) # if it doesn't move it dies
            
    return game    


def init(game, num=2500):
    num = min(num, len(empty_positions(game)))
    for i in range(num):
        new_cell = random_spawn(game)
        if len(game.cells) == 0:
            game.add_cell(*new_cell)
        else:
            cell = random.choice(game.cells)
            game.add_cell(*new_cell, genes=game.mutate(cell))

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
            game = init(Game(size=args.size), num=args.initial_cells)

        countdown = ROUNDTIME
        frame = 0
        while args.max_frames is None or frame < args.max_frames:
            try:
                if len(game.cells) == 0:
                    game = init(game, num=MIN_WAVE)
                if args.snapshot_dir and frame % args.snapshot_every == 0:
                    write_snapshot(game, countdown, frame, args.snapshot_dir)
                if not args.no_render:
                    cls()
                    print_grid(game)
                    print(status_line(game, countdown))
                game, countdown = advance_round(game, countdown)

                if args.frame_rate > 0:
                    time.sleep(args.frame_rate)
                game = step(game)
                countdown -= 1
                frame += 1

            except KeyboardInterrupt:
                print('Saving...')
                save_state(game, args.save)
                print('Saved!')
                break

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
    parser.add_argument('--max-frames', type=positive_int, help='stop after this many frames')
    parser.add_argument('--snapshot-dir', help='write plain-text frame snapshots to this directory')
    parser.add_argument('--snapshot-every', type=positive_int, default=1, help='write one snapshot every N frames')
    parser.add_argument('--no-render', action='store_true', help='do not render frames to the terminal')
    parser.add_argument('--frame-rate', type=float, default=FRAME_RATE, help='seconds to sleep between frames')
    args = parser.parse_args()

    main(args)
