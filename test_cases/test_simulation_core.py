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


def test_structured_matrix_noise_is_low_rank():
    npd.seed_all(123)
    matrix = torch.zeros(9, 33)

    noise = npd.structured_noise_like(matrix, scale=1.0, rank=2)

    assert noise.shape == matrix.shape
    assert not torch.allclose(noise, torch.zeros_like(noise))
    assert torch.linalg.matrix_rank(noise, tol=1e-5).item() <= 2


def test_light_mutation_preserves_original_second_random_draw(monkeypatch):
    cell = npd.Cell([3, 3])
    game = npd.Game(size=(6, 6))
    rolls = iter([0.5, 0.0])
    monkeypatch.setattr(npd.np.random, 'rand', lambda: next(rolls))
    monkeypatch.setattr(npd.torch, 'randn_like', lambda tensor: torch.ones_like(tensor))

    genes = game.mutate(cell)

    matrix_delta = 0.00001 * npd.LOW_RANK_MUTATION_RANK ** 0.5
    assert torch.allclose(genes['weight_1'], cell.linear.weight + matrix_delta)
    assert torch.allclose(genes['bias_1'], cell.linear.bias + 0.00001)
    assert torch.allclose(genes['weight_2'], cell.linear2.weight + matrix_delta)
    assert torch.allclose(genes['bias_2'], cell.linear2.bias + 0.00001)


def test_legacy_mutation_mode_uses_dense_noise(monkeypatch):
    cell = npd.Cell([3, 3])
    game = npd.Game(size=(6, 6), mutation_mode=npd.MUTATION_MODE_LEGACY)
    rolls = iter([0.5, 0.0])
    monkeypatch.setattr(npd.np.random, 'rand', lambda: next(rolls))
    monkeypatch.setattr(npd.torch, 'randn_like', lambda tensor: torch.ones_like(tensor))

    genes = game.mutate(cell)

    assert torch.allclose(genes['weight_1'], cell.linear.weight + 0.00001)
    assert torch.allclose(genes['bias_1'], cell.linear.bias + 0.00001)
    assert torch.allclose(genes['weight_2'], cell.linear2.weight + 0.00001)
    assert torch.allclose(genes['bias_2'], cell.linear2.bias + 0.00001)


def test_shared_rank1_spawn_uses_one_base_per_wave():
    npd.seed_all(7)
    game = npd.Game(size=(6, 6), mutation_mode=npd.MUTATION_MODE_SHARED_RANK1)

    npd.init(game, num=4)

    assert len(game.cells) == 4
    assert {cell.genome_mode for cell in game.cells} == {npd.MUTATION_MODE_SHARED_RANK1}
    assert {cell.base_id for cell in game.cells} == {0}
    for cell in game.cells:
        assert torch.allclose(cell.shared_base['weight_1'], game.shared_base_genes['weight_1'])
        assert torch.allclose(cell.linear.weight, cell.shared_base['weight_1'] + npd.rank1_delta(
            cell.rank1['weight_1_u'],
            cell.rank1['weight_1_v'],
        ))


def test_shared_rank1_base_updates_from_hp_weighted_survivors():
    game = npd.Game(size=(6, 6), mutation_mode=npd.MUTATION_MODE_SHARED_RANK1)
    game.add_cell(2, 2)
    game.add_cell(2, 3)
    first, second = game.cells
    first.health = 1
    second.health = 3

    with torch.no_grad():
        first.linear.weight.fill_(2)
        second.linear.weight.fill_(10)

    game.update_shared_base_from_survivors()

    assert torch.allclose(game.shared_base_genes['weight_1'], torch.full_like(first.linear.weight, 8.0))
    assert game.shared_base_id == 1


def test_batched_shared_rank1_actions_match_materialized_networks():
    npd.seed_all(11)
    game = npd.Game(size=(6, 6), mutation_mode=npd.MUTATION_MODE_SHARED_RANK1)
    npd.init(game, num=3)
    cells = game.cells
    neighbors = [torch.randn(24) for _cell in cells]
    expected = []

    for cell, neighbor in zip(cells, neighbors):
        inputs = torch.cat((neighbor, torch.zeros(9))).unsqueeze(0)
        logits = cell.linear2(torch.relu(cell.linear(inputs)))
        expected.append(logits.argmax().item())

    assert npd.batched_shared_rank1_actions(cells, neighbors) == expected


def test_simultaneous_attack_can_catch_moving_target(monkeypatch):
    game = npd.Game(size=(8, 8), action_mode=npd.ACTION_MODE_SIMULTANEOUS)
    game.add_cell(4, 4)
    game.add_cell(4, 5)
    attacker = game.get_cell(4, 4)
    victim = game.get_cell(4, 5)
    force_action(attacker, 3)
    force_action(victim, 3)
    victim.health = 5
    monkeypatch.setattr(npd.random, 'choice', lambda items: items[0])
    monkeypatch.setattr(npd.random, 'random', lambda: 0.0)

    npd.step(game)

    assert victim not in game.cells
    assert attacker in game.cells
    assert attacker.pos == [4, 5]
    assert game.get_cell(4, 4) is not False


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


def test_video_renderer_writes_artifact_and_manifest(tmp_path):
    script = Path(__file__).resolve().parent / 'render_video.py'
    output = tmp_path / 'preview.mp4'
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            '--output',
            str(output),
            '--render-rounds',
            '2',
            '--round-stride',
            '2',
            '--fps',
            '10',
            '--size',
            '5x5',
            '--initial-cells',
            '4',
            '--seed',
            '7',
            '--cell-size',
            '4',
            '--write-manifest',
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert output.stat().st_size > 0
    manifest = output.with_suffix(output.suffix + '.manifest.txt')
    assert manifest.exists()
    manifest_text = manifest.read_text(encoding='utf-8')
    assert 'render_rounds: 2' in manifest_text
    assert 'round_stride: 2' in manifest_text
    assert 'frames_written: 1001' in manifest_text
    assert 'rendered_rounds: 0,2' in manifest_text


def test_mutation_comparison_writes_csv_plot_and_videos(tmp_path):
    script = Path(__file__).resolve().parent / 'compare_mutation_modes.py'
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            '--output-dir',
            str(tmp_path),
            '--rounds',
            '2',
            '--render-every-rounds',
            '1',
            '--roundtime',
            '3',
            '--fps',
            '10',
            '--size',
            '5x5',
            '--initial-cells',
            '4',
            '--seed',
            '7',
            '--cell-size',
            '4',
            '--action-mode',
            'simultaneous',
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'survival_rounds_2.csv').exists()
    assert (tmp_path / 'survival_rounds_2.png').exists()
    for mode in npd.MUTATION_MODES:
        video = tmp_path / f'{mode}_rounds_2_every_1.mp4'
        manifest = video.with_suffix(video.suffix + '.manifest.txt')
        assert video.exists()
        assert video.stat().st_size > 0
        manifest_text = manifest.read_text(encoding='utf-8')
        assert f'mutation_mode: {mode}' in manifest_text
        assert 'action_mode: simultaneous' in manifest_text

    csv_text = (tmp_path / 'survival_rounds_2.csv').read_text(encoding='utf-8')
    assert 'mutation_mode,round,previous_round_cells,pre_refill_cells,post_refill_cells' in csv_text
    assert 'legacy,0,' in csv_text
    assert 'low_rank,0,' in csv_text
