import numpy as np
from numpy.lib.function_base import average, select
from numpy.random import rand
import torch
from torch._C import Argument
import torch.nn as nn
import torch.nn.functional as F
import os
from sty import fg, bg, ef, rs
import time
import random
import argparse
import pickle

import signal
import sys
# Weird errors where the cells can be in the same location and not die, and cells can move around in a weird way that constantly gives them health points

cls = lambda: os.system('clear') or None
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
        self.pos = pos
        self.health = 2
        self.max_health = 15
        self.age = 0
        self.prev_state = None
        self.diversity = None

        self.pos_list = []

        if genes != None:
            self.linear.weight = nn.Parameter(genes['weight_1'])
            self.linear.bias = nn.Parameter(genes['bias_1'])
            self.linear2.weight = nn.Parameter(genes['weight_2'])
            self.linear2.bias = nn.Parameter(genes['bias_2'])

    def forward(self, neighbors):
        inps = neighbors.unsqueeze(0)
        if self.prev_state != None:
            all_imps = torch.cat((inps, self.prev_state), dim=-1)
        else:
            all_imps = torch.cat((inps, torch.zeros(1, 1, 9)), dim=-1)
        lin1 = self.relu(self.linear(all_imps))
        self.prev_state = lin1
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
        self.pos = pos if str(type(pos)) != "<class 'numpy.ndarray'>" else pos.tolist()
                    
            

    def get_genes(self):
        return {
            'weight_1': self.linear.weight, 
            'bias_1': self.linear.bias,
            'weight_2': self.linear2.weight,
            'bias_2': self.linear2.bias
            }

    def add_health(self, amount=1):
        if self.health < self.max_health:
            if amount > 1:
                amount = amount if amount + self.health <= self.max_health else self.max_health - self.health
            self.health += amount

    def total_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def random_spawn(game):
    '''returns a random position in the grid that is not occupied'''
    while True:
        pos = np.array([np.random.randint(2, game.size.lines), np.random.randint(2, game.size.columns+2)])
        if game.grid[pos[0], pos[1]] == 0:
            return pos

class Game():
    '''
    Manages Game State
    '''
    def __init__(self, genepool=None):
        self.size = os.get_terminal_size()
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
        if [*new] == [20, 213]:
            raise Exception('fuck')
        if cell == None:
            cell = self.get_cell(y, x)
            if cell == False:
                raise Exception('Cell Does not Exist at this Position')
       
        cell.update_pos([new[0], new[1]])
        self.grid[new[0], new[1]] = 1
        self.grid[y][x] = 0

    def remove_cell(self, y, x): 
        #self.graveyard.append(self.get_cell(y, x)) # change so cell is passed in
        self.grid[y, x] = 0
        self.cells = [c for c in self.cells if c.pos != [y, x]]

    def damage_cell(self, cell):
        cell.health -= 1
        if cell.health == 0:
            self.remove_cell(*cell.pos)
            return True
        else:
            False

    def add_cell(self, y, x, genes=None):
        self.grid[y][x] = 1
        self.cells.append(Cell([y, x], genes))

    def mutate(self, cell):
        '''
        Mutates a cell's genes bashed on the surrounding cells
        '''
        # 10% chance of mutation
        
        if np.random.rand() < 0.05:
            # get the surrounding positions
            neighbors = [cell.pos + direction_dict[i](cell.pos) for i in range(1,9)] # wow clever copilot
            ncells = [self.get_cell(*n) for n in neighbors]
            ncells = [c for c in ncells if c != False]
            if len(ncells) != 0:
                genes = [c.get_genes() for c in ncells]
                # combine the genes 
                new_genes = {}
                for k in genes[0].keys():
                    new_genes[k] = torch.mean(torch.stack([g[k] for g in genes]), axis=0)
                # alter the genes of the cell through linear combination and add random noise with a small guassian
                weight1 = nn.Parameter((cell.linear.weight * 0.8 + new_genes['weight_1'] * 0.2) + torch.randn(cell.linear.weight.shape) * 0.1)
                bias1 = nn.Parameter((cell.linear.bias * 0.8 + new_genes['bias_1'] * 0.2) + torch.randn(cell.linear.bias.shape) * 0.1)
                weight2 = nn.Parameter((cell.linear2.weight * 0.8 + new_genes['weight_2'] * 0.2) + torch.randn(cell.linear2.weight.shape) * 0.1)
                bias2 = nn.Parameter((cell.linear2.bias * 0.8 + new_genes['bias_2'] * 0.2) + torch.randn(cell.linear2.bias.shape) * 0.1)
            else:
                weight1 = nn.Parameter(cell.linear.weight + torch.randn(cell.linear.weight.shape) * 0.001)
                bias1 = nn.Parameter(cell.linear.bias + torch.randn(cell.linear.bias.shape) * 0.001)   
                weight2 = nn.Parameter(cell.linear2.weight + torch.randn(cell.linear2.weight.shape) * 0.001)
                bias2 = nn.Parameter(cell.linear2.bias + torch.randn(cell.linear2.bias.shape) * 0.001)
        elif np.random.rand() < 0.4:
            weight1 = cell.linear.weight + torch.randn(cell.linear.weight.shape) * 0.00001
            bias1 = cell.linear.bias + torch.randn(cell.linear.bias.shape) * 0.00001
            weight2 = cell.linear2.weight + torch.randn(cell.linear2.weight.shape) * 0.00001
            bias2 = cell.linear2.bias  + torch.randn(cell.linear2.bias.shape) * 0.00001
        else:
            # no mutation
            weight1 = cell.linear.weight
            bias1 = cell.linear.bias
            weight2 = cell.linear2.weight
            bias2 = cell.linear2.bias
        return {'weight_1': weight1, 'bias_1': bias1,
                'weight_2': weight2, 'bias_2': bias2}


def print_grid(game):
    # skip the 2 layer border
    for row in range(2, game.size.lines):
        pstring = [BLANK]*game.size.columns
        cells_row = game.grid[row]
        #if sum(cells_row) != 0: # doesn't account for padding
        for col in range(2, game.size.columns + 2):
            if cells_row[col] == 1:
                color = int((87 - 4*np.sum(game.grid[row-1:row+2,col-1:col+2])) % 255) # color is based on density of cells # could remove -1's to account for border padding
                pstring[col - 2] = f'{fg(color)}{X}{fg.rs}' # -2 to account for skipped iteration from the two layer border
        print(''.join(pstring))


def step(game):
    for cell in game.cells:
        y, x = cell.pos
        # get the surrounding positions
        neighbors = np.delete(game.grid[y-2:y+3, x-2:x+3].reshape(-1), 12, 0) 
        if neighbors.shape[0] != 24:
            print(game.grid[y-2:y+3, x-2:x+3])
            print(game.grid[20, 213])
        action = cell(torch.tensor([neighbors], dtype=torch.float32)).int().tolist()

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
    total_in_game = len(game.cells)
    for i in range(num):
        new_cell = random_spawn(game)
        if total_in_game == 0:
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
    with torch.no_grad():
        if args.load:
            game = pickle.load(open(args.load, 'rb'))
        else:
            game = init(Game())

        countdown = ROUNDTIME
        while True:
            try:
                cls()
                print_grid(game)
                avghealth = round(sum([c.health for c in game.cells]) / len(game.cells))
                maxhealth = max([c.health for c in game.cells])
                maxage = max([c.age for c in game.cells])
                totalcells = len(game.cells)
                print(f'{fg.red}Petri Dish{fg.rs}{fg.green}      AVG HP: {avghealth}   MAX HP: {maxhealth}{fg.rs}    {fg.white}Total Players: {totalcells}{fg.rs}   {fg.red}Oldest Cell: {maxage}   Total Rounds: {game.rounds}{fg.rs}    {fg.blue}Countdown: {countdown}{fg.rs}')
                if countdown == 0:
                    game = init(game, num=max(PER_WAVE - totalcells, MIN_WAVE))
                    countdown = ROUNDTIME
                    game.rounds += 1

                    if totalcells > MAX_TOTAL:
                        game = prune(game) # prune the game if it is too big
                    for cell in game.cells:
                        cell.age += 1
                        if cell.age == maxage:
                            cell.add_health()

                time.sleep(FRAME_RATE)
                game = step(game)
                countdown -= 1

            except KeyboardInterrupt:
                print('Saving...')
                save_state(game, args.save)
                print('Saved!')
                break

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
    args = parser.parse_args()

    main(args)
