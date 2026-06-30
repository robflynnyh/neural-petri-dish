import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import neural_petri_dish as npd


def force_action(cell, action):
    with torch.no_grad():
        cell.linear.weight.zero_()
        cell.linear.bias.zero_()
        cell.linear2.weight.zero_()
        cell.linear2.bias.fill_(-1)
        cell.linear2.bias[action] = 1


def positions(game):
    return sorted(tuple(cell.pos) for cell in game.cells)


def test_game_can_be_created_without_real_terminal():
    game = npd.Game()

    assert game.size.lines > 2
    assert game.size.columns > 0
    assert game.grid.shape == (game.size.lines + 2, game.size.columns + 4)


def test_init_caps_spawn_count_to_available_positions():
    game = npd.Game(size=(4, 5))

    npd.init(game, num=999)

    assert len(game.cells) == 10
    assert len(positions(game)) == 10
    assert len(npd.empty_positions(game)) == 0


def test_initial_empty_game_spawn_keeps_random_independent_genes(monkeypatch):
    game = npd.Game(size=(6, 6))

    def fail_mutate(_game, _cell):
        raise AssertionError('empty initial spawn should not mutate from the first cell')

    monkeypatch.setattr(npd.Game, 'mutate', fail_mutate)

    npd.init(game, num=8)

    assert len(game.cells) == 8


def test_wave_spawn_mutates_only_preexisting_cell_count(monkeypatch):
    game = npd.Game(size=(6, 6))
    game.add_cell(2, 2)
    mutate_calls = []

    def count_mutate(_game, cell):
        mutate_calls.append(cell)
        return cell.get_genes()

    monkeypatch.setattr(npd.Game, 'mutate', count_mutate)

    npd.init(game, num=5)

    assert len(game.cells) == 6
    assert len(mutate_calls) == 1


def test_random_spawn_raises_when_grid_is_full():
    game = npd.Game(size=(3, 3))
    npd.init(game, num=999)

    with pytest.raises(RuntimeError, match='No Empty Positions Available'):
        npd.random_spawn(game)


def test_random_spawn_preserves_coordinate_draw_path(monkeypatch):
    game = npd.Game(size=(6, 6))
    draws = iter([3, 4])
    monkeypatch.setattr(npd.np.random, 'randint', lambda *_args: next(draws))

    pos = npd.random_spawn(game)

    assert pos.tolist() == [3, 4]


def test_add_cell_rejects_occupied_position():
    game = npd.Game(size=(6, 6))
    game.add_cell(3, 3)

    with pytest.raises(Exception, match='Cannot Add Cell'):
        game.add_cell(3, 3)


def test_update_cell_rejects_occupied_position():
    game = npd.Game(size=(6, 6))
    game.add_cell(3, 3)
    game.add_cell(3, 4)

    with pytest.raises(Exception, match='Occupied'):
        game.update_cell(3, 3, [3, 4])


def test_step_skips_cell_removed_earlier_in_same_tick(monkeypatch):
    game = npd.Game(size=(8, 8))
    game.add_cell(4, 4)
    game.add_cell(4, 5)
    attacker = game.get_cell(4, 4)
    victim = game.get_cell(4, 5)
    victim.health = 1
    force_action(attacker, 3)

    def fail_if_called(_neighbors):
        raise AssertionError('removed victim should not act')

    monkeypatch.setattr(victim, 'forward', fail_if_called)

    npd.step(game)

    assert len(game.cells) == 2
    assert positions(game) == [(4, 4), (4, 5)]
    assert game.get_cell(4, 5) is attacker


def test_unsuccessful_attack_damages_attacker_and_victim():
    game = npd.Game(size=(8, 8))
    game.add_cell(4, 4)
    game.add_cell(4, 5)
    attacker = game.get_cell(4, 4)
    victim = game.get_cell(4, 5)
    force_action(attacker, 3)
    force_action(victim, 0)

    npd.step(game)

    assert len(game.cells) == 2
    assert attacker.health == 1
    assert victim.health == 1
    assert positions(game) == [(4, 4), (4, 5)]


def test_mutation_uses_actual_neighbor_positions(monkeypatch):
    game = npd.Game(size=(8, 8))
    game.add_cell(4, 4)
    game.add_cell(4, 5)
    parent = game.get_cell(4, 4)
    neighbor = game.get_cell(4, 5)

    with torch.no_grad():
        parent.linear.weight.zero_()
        parent.linear.bias.zero_()
        parent.linear2.weight.zero_()
        parent.linear2.bias.zero_()
        neighbor.linear.weight.fill_(10)
        neighbor.linear.bias.fill_(20)
        neighbor.linear2.weight.fill_(30)
        neighbor.linear2.bias.fill_(40)

    monkeypatch.setattr(npd.np.random, 'rand', lambda: 0.0)
    monkeypatch.setattr(npd.torch, 'randn_like', lambda tensor: torch.zeros_like(tensor))

    genes = game.mutate(parent)

    assert torch.allclose(genes['weight_1'], torch.full_like(parent.linear.weight, 2.0))
    assert torch.allclose(genes['bias_1'], torch.full_like(parent.linear.bias, 4.0))
    assert torch.allclose(genes['weight_2'], torch.full_like(parent.linear2.weight, 6.0))
    assert torch.allclose(genes['bias_2'], torch.full_like(parent.linear2.bias, 8.0))


def test_light_mutation_preserves_original_second_random_draw(monkeypatch):
    cell = npd.Cell([3, 3])
    game = npd.Game(size=(6, 6))
    rolls = iter([0.5, 0.0])
    monkeypatch.setattr(npd.np.random, 'rand', lambda: next(rolls))
    monkeypatch.setattr(npd.torch, 'randn_like', lambda tensor: torch.ones_like(tensor))

    genes = game.mutate(cell)

    assert torch.allclose(genes['weight_1'], cell.linear.weight + 0.00001)
    assert torch.allclose(genes['bias_1'], cell.linear.bias + 0.00001)
    assert torch.allclose(genes['weight_2'], cell.linear2.weight + 0.00001)
    assert torch.allclose(genes['bias_2'], cell.linear2.bias + 0.00001)


def test_cell_genes_are_cloned_not_shared():
    parent = npd.Cell([3, 3])
    child = npd.Cell([3, 4], parent.get_genes())

    with torch.no_grad():
        parent.linear.weight.add_(100)

    assert not torch.allclose(parent.linear.weight, child.linear.weight)


def test_cli_bounded_run_writes_snapshots(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'neural_petri_dish.py'
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            '--max-frames',
            '6',
            '--snapshot-every',
            '2',
            '--snapshot-dir',
            str(tmp_path),
            '--no-render',
            '--frame-rate',
            '0',
            '--size',
            '8x12',
            '--initial-cells',
            '10',
            '--seed',
            '7',
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    snapshots = sorted(tmp_path.glob('frame_*.txt'))
    assert [path.name for path in snapshots] == [
        'frame_00000.txt',
        'frame_00002.txt',
        'frame_00004.txt',
    ]
    first_snapshot = snapshots[0].read_text(encoding='utf-8')
    assert 'Frame: 0' in first_snapshot
    assert 'Petri Dish' in first_snapshot
    assert '#' in first_snapshot
