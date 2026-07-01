import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import neural_petri_dish as npd


def force_action(cell, action, attack=False):
    with torch.no_grad():
        cell.linear.weight.zero_()
        cell.linear.bias.zero_()
        cell.linear2.weight.zero_()
        cell.linear2.bias.fill_(-1)
        cell.linear2.bias[action] = 1
        cell.linear2.bias[npd.ATTACK_OUTPUT_INDEX] = 1 if attack else -1


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


def test_initial_empty_game_spawn_uses_one_rank1_family(monkeypatch):
    game = npd.Game(size=(6, 6))

    def fail_mutate(_game, _cell):
        raise AssertionError('initial rank-1 wave should not mutate from an existing cell')

    monkeypatch.setattr(npd.Game, 'mutate', fail_mutate)

    npd.init(game, num=8)

    assert len(game.cells) == 8
    assert len({cell.rank1_family for cell in game.cells}) == 1


def test_wave_spawn_uses_hp_weighted_family_not_per_cell_mutate(monkeypatch):
    game = npd.Game(size=(6, 6))
    game.add_cell(2, 2)
    survivor = game.get_cell(2, 2)
    survivor.rank1_family = npd.SharedRank1Family(survivor.get_genes())

    def fail_mutate(_game, _cell):
        raise AssertionError('rank-1 refill waves should come from one weighted family')

    monkeypatch.setattr(npd.Game, 'mutate', fail_mutate)

    npd.init(game, num=5)

    assert len(game.cells) == 6
    new_cells = [cell for cell in game.cells if cell is not survivor]
    assert len({cell.rank1_family for cell in new_cells}) == 1


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
    force_action(attacker, 3, attack=True)

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
    game.add_cell(5, 5)
    attacker = game.get_cell(4, 4)
    victim = game.get_cell(4, 5)
    protector = game.get_cell(5, 5)
    force_action(attacker, 3, attack=True)
    force_action(victim, 0)
    force_action(protector, 0)

    npd.step(game)

    assert len(game.cells) == 3
    assert attacker.health == 1
    assert victim.health == 1
    assert protector.health == 2
    assert positions(game) == [(4, 4), (4, 5), (5, 5)]


def test_move_intent_into_occupied_square_misses_turn():
    game = npd.Game(size=(8, 8))
    game.add_cell(4, 4)
    game.add_cell(4, 5)
    mover = game.get_cell(4, 4)
    blocker = game.get_cell(4, 5)
    force_action(mover, 3, attack=False)
    force_action(blocker, 0)

    npd.step(game)

    assert mover.health == 2
    assert blocker.health == 2
    assert mover.pos == [4, 4]
    assert blocker.pos == [4, 5]


def test_successful_move_costs_fractional_health():
    game = npd.Game(size=(8, 8))
    game.add_cell(4, 4)
    mover = game.get_cell(4, 4)
    mover.health = 2.0
    force_action(mover, 3, attack=False)

    npd.step(game)

    assert len(game.cells) == 1
    assert mover.pos == [4, 5]
    assert mover.health == pytest.approx(1.9)


def test_successful_move_can_kill_without_health_floor():
    game = npd.Game(size=(8, 8))
    game.add_cell(4, 4)
    mover = game.get_cell(4, 4)
    mover.health = 0.05
    force_action(mover, 3, attack=False)

    npd.step(game)

    assert mover not in game.cells
    assert game.grid[4, 5] == 0


def test_attack_intent_into_empty_square_misses_turn():
    game = npd.Game(size=(8, 8))
    game.add_cell(4, 4)
    attacker = game.get_cell(4, 4)
    force_action(attacker, 3, attack=True)

    npd.step(game)

    assert len(game.cells) == 1
    assert attacker.health == 2
    assert attacker.pos == [4, 4]


def test_stationary_action_damages_and_can_kill():
    game = npd.Game(size=(8, 8))
    game.add_cell(4, 4)
    cell = game.get_cell(4, 4)
    cell.health = 2.0
    force_action(cell, 0)

    for _ in range(4):
        npd.step(game)

    assert cell not in game.cells
    assert game.grid[4, 4] == 0


def test_lone_target_takes_extra_attack_damage():
    game = npd.Game(size=(8, 8))
    game.add_cell(4, 4)
    game.add_cell(4, 5)
    attacker = game.get_cell(4, 4)
    victim = game.get_cell(4, 5)
    force_action(attacker, 3, attack=True)
    force_action(victim, 0)

    npd.step(game)

    assert victim not in game.cells
    assert attacker in game.cells
    assert attacker.pos == [4, 5]
    assert attacker.health == 7.0


def test_kill_reward_is_capped_at_max_health():
    game = npd.Game(size=(8, 8))
    game.add_cell(4, 4)
    game.add_cell(4, 5)
    attacker = game.get_cell(4, 4)
    victim = game.get_cell(4, 5)
    attacker.health = 14.0
    victim.health = 1.0
    force_action(attacker, 3, attack=True)
    force_action(victim, 0)

    npd.step(game)

    assert attacker.health == 15.0


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
    monkeypatch.setattr(npd.np.random, 'randn', lambda: 0.0)
    monkeypatch.setattr(npd.torch, 'randn_like', lambda tensor: torch.zeros_like(tensor))

    genes = game.mutate(parent)

    assert torch.allclose(genes['weight_1'], torch.full_like(parent.linear.weight, 2.0))
    assert torch.allclose(genes['bias_1'], torch.full_like(parent.linear.bias, 4.0))
    assert torch.allclose(genes['weight_2'], torch.full_like(parent.linear2.weight, 6.0))
    assert torch.allclose(genes['bias_2'], torch.full_like(parent.linear2.bias, 8.0))
    assert torch.allclose(genes['_rank1_family'].base_weight_1, torch.full_like(parent.linear.weight, 2.0))
    assert torch.allclose(genes['_rank1_family'].base_weight_2, torch.full_like(parent.linear2.weight, 6.0))


def test_shared_rank1_family_direction_is_rank1_with_controlled_scale():
    npd.seed_all(123)
    family = npd.SharedRank1Family()

    direction = torch.outer(family.u_1, family.v_1)

    assert direction.shape == family.base_weight_1.shape
    assert not torch.allclose(direction, torch.zeros_like(direction))
    singular_values = torch.linalg.svdvals(direction)
    assert singular_values[1:].max() / singular_values[0] < 1e-5
    assert torch.isclose(direction.square().mean().sqrt(), torch.tensor(1.0), atol=1e-5)


def test_factored_initial_wave_uses_one_compatible_family():
    npd.seed_all(123)
    game = npd.Game(size=(8, 8), mutation_mode=npd.MUTATION_MODE_SHARED_RANK1_FACTORED)

    npd.init(game, num=8)

    families = {id(cell.rank1_family) for cell in game.cells}
    assert len(families) == 1
    for cell in game.cells:
        expected_weight_1 = cell.rank1_family.materialize_weight_1(cell.rank1_coeff_1)
        expected_weight_2 = cell.rank1_family.materialize_weight_2(cell.rank1_coeff_2)
        assert torch.allclose(cell.linear.weight, expected_weight_1)
        assert torch.allclose(cell.linear2.weight, expected_weight_2)


def test_factored_wave_cells_share_family_base_biases():
    npd.seed_all(123)
    game = npd.Game(size=(8, 8), mutation_mode=npd.MUTATION_MODE_SHARED_RANK1_FACTORED)

    npd.init(game, num=8)

    family = game.cells[0].rank1_family
    assert all(cell.linear.bias is family.base_bias_1 for cell in game.cells)
    assert all(cell.linear2.bias is family.base_bias_2 for cell in game.cells)


def test_factored_refill_wave_creates_new_family_without_reassigning_survivors():
    npd.seed_all(123)
    game = npd.Game(size=(10, 10), mutation_mode=npd.MUTATION_MODE_SHARED_RANK1_FACTORED)
    npd.init(game, num=4)
    survivors = list(game.cells)
    survivor_family = survivors[0].rank1_family
    for index, cell in enumerate(survivors, start=1):
        cell.health = index
    weights = torch.tensor([cell.health for cell in survivors], dtype=torch.float32)
    weights = weights / weights.sum()
    expected_base_weight_1 = sum(cell.linear.weight * weights[index] for index, cell in enumerate(survivors))
    expected_base_weight_2 = sum(cell.linear2.weight * weights[index] for index, cell in enumerate(survivors))

    npd.init(game, num=4)

    assert {cell.rank1_family for cell in survivors} == {survivor_family}
    refill_cells = [cell for cell in game.cells if cell not in survivors]
    assert len(refill_cells) == 4
    refill_families = {cell.rank1_family for cell in refill_cells}
    assert len(refill_families) == 1
    assert survivor_family not in refill_families
    refill_family = refill_cells[0].rank1_family
    assert torch.allclose(refill_family.base_weight_1, expected_base_weight_1)
    assert torch.allclose(refill_family.base_weight_2, expected_base_weight_2)


def test_family_batched_actions_match_dense_cell_forward():
    npd.seed_all(123)
    game = npd.Game(
        size=(12, 12),
        mutation_mode=npd.MUTATION_MODE_SHARED_RANK1_FACTORED,
        action_backend=npd.ACTION_BACKEND_FAMILY_BATCHED,
        action_device='cpu',
        batched_min_family_size=1,
    )
    npd.init(game, num=12)
    cells = list(game.cells)

    planned = npd.planned_family_actions(game, cells)

    assert len(planned) == len(cells)
    for cell in cells:
        y, x = cell.pos
        neighbors = game.grid[y-2:y+3, x-2:x+3].reshape(-1)
        dense_action = cell.forward_neighbors25(neighbors)
        batched_action, batched_hidden = planned[id(cell)]
        assert batched_action == dense_action
        assert npd.np.allclose(batched_hidden, cell.prev_state, atol=1e-6)


def test_packed_family_actions_match_dense_cell_forward_with_multiple_families():
    npd.seed_all(456)
    game = npd.Game(
        size=(12, 12),
        mutation_mode=npd.MUTATION_MODE_SHARED_RANK1_FACTORED,
        action_backend=npd.ACTION_BACKEND_FAMILY_BATCHED,
        action_device='cpu',
        batched_min_family_size=1,
    )
    npd.init(game, num=12)
    cells = list(game.cells)
    for cell in cells[:4]:
        cell.rank1_family = npd.SharedRank1Family(cell.get_genes())
        cell.rank1_coeff_1 = 0.0
        cell.rank1_coeff_2 = 0.0

    planned = npd.planned_family_action_list(game, cells)

    assert len(planned) == len(cells)
    assert len({cell.rank1_family for cell in cells}) > 1
    for planned_index, cell in enumerate(cells):
        y, x = cell.pos
        neighbors = game.grid[y-2:y+3, x-2:x+3].reshape(-1)
        dense_action = cell.forward_neighbors25(neighbors)
        packed_action, packed_hidden = planned[planned_index]
        assert packed_action == dense_action
        assert npd.np.allclose(packed_hidden, cell.prev_state, atol=1e-6)


def test_light_rank1_mutation_keeps_family_and_moves_coefficients(monkeypatch):
    cell = npd.Cell([3, 3])
    family = npd.SharedRank1Family(cell.get_genes())
    cell.rank1_family = family
    game = npd.Game(size=(6, 6))
    rolls = iter([0.5, 0.0])
    monkeypatch.setattr(npd.np.random, 'rand', lambda: next(rolls))
    monkeypatch.setattr(npd.np.random, 'randn', lambda: 1.0)
    monkeypatch.setattr(npd.torch, 'randn_like', lambda tensor: torch.ones_like(tensor))

    genes = game.mutate(cell)

    assert genes['_rank1_family'] is family
    assert genes['_rank1_coeff_1'] == cell.rank1_coeff_1 + game.mutate_rate
    assert genes['_rank1_coeff_2'] == cell.rank1_coeff_2 + game.mutate_rate
    assert torch.allclose(genes['weight_1'], family.materialize_weight_1(genes['_rank1_coeff_1']))
    assert torch.allclose(genes['bias_1'], cell.linear.bias + game.mutate_rate)


def test_shared_rank1_spawn_uses_one_base_per_wave():
    npd.seed_all(7)
    game = npd.Game(size=(6, 6), mutation_mode=npd.MUTATION_MODE_SHARED_RANK1_FACTORED)

    npd.init(game, num=4)

    assert len(game.cells) == 4
    assert len({cell.rank1_family for cell in game.cells}) == 1
    for cell in game.cells:
        family = cell.rank1_family
        assert torch.allclose(cell.linear.weight, family.materialize_weight_1(cell.rank1_coeff_1))
        assert torch.allclose(cell.linear2.weight, family.materialize_weight_2(cell.rank1_coeff_2))


def test_shared_rank1_base_updates_from_hp_weighted_survivors():
    game = npd.Game(size=(6, 6), mutation_mode=npd.MUTATION_MODE_SHARED_RANK1_FACTORED)
    game.add_cell(2, 2)
    game.add_cell(2, 3)
    first, second = game.cells
    first.health = 1
    second.health = 3

    with torch.no_grad():
        first.linear.weight.fill_(2)
        second.linear.weight.fill_(10)
        first.linear2.weight.fill_(4)
        second.linear2.weight.fill_(12)

    family = game.make_factored_wave_family()

    assert torch.allclose(family.base_weight_1, torch.full_like(first.linear.weight, 8.0))
    assert torch.allclose(family.base_weight_2, torch.full_like(first.linear2.weight, 10.0))


def test_round_transition_does_not_cost_survivor_health_before_wave_spawn():
    game = npd.Game(size=(6, 6), mutation_mode=npd.MUTATION_MODE_SHARED_RANK1_FACTORED)
    game.add_cell(2, 2)
    game.add_cell(2, 3)
    first, second = game.cells
    first.health = 1
    second.health = 2

    game, countdown = npd.advance_round(game, 0)

    assert countdown == npd.ROUNDTIME
    assert game.rounds == 1
    assert first in game.cells
    assert second in game.cells
    assert first.health == 1
    assert second.health == 2


def test_batched_shared_rank1_actions_match_materialized_networks():
    npd.seed_all(11)
    game = npd.Game(
        size=(6, 6),
        mutation_mode=npd.MUTATION_MODE_SHARED_RANK1_FACTORED,
        action_backend=npd.ACTION_BACKEND_FAMILY_BATCHED,
        action_device='cpu',
        batched_min_family_size=1,
    )
    npd.init(game, num=3)
    cells = game.cells
    prev_states = [cell.prev_state.copy() for cell in cells]
    expected = []

    for cell in cells:
        y, x = cell.pos
        neighbors = game.grid[y-2:y+3, x-2:x+3].reshape(25)
        expected.append(cell.forward_flat_neighbors25(neighbors))

    for cell, prev_state in zip(cells, prev_states):
        cell.prev_state[:] = prev_state
    planned = npd.planned_family_action_list(game, cells)
    assert [action for action, _hidden in planned] == expected


def test_lethal_attack_moves_attacker_and_refills_origin(monkeypatch):
    game = npd.Game(size=(8, 8))
    game.add_cell(4, 4)
    game.add_cell(4, 5)
    attacker = game.get_cell(4, 4)
    victim = game.get_cell(4, 5)
    force_action(attacker, 3, attack=True)
    force_action(victim, 3)
    victim.health = 1
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


def test_cli_tensor_engine_bounded_run_writes_snapshots(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'neural_petri_dish.py'
    metrics_path = tmp_path / 'metrics.json'
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            '--engine',
            npd.SIM_ENGINE_TENSOR_RANK1,
            '--action-device',
            'cpu',
            '--max-frames',
            '6',
            '--tensor-block-steps',
            '2',
            '--snapshot-every',
            '2',
            '--snapshot-dir',
            str(tmp_path),
            '--metrics-json',
            str(metrics_path),
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
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    assert metrics['engine'] == npd.SIM_ENGINE_TENSOR_RANK1
    assert metrics['frames'] == 6
    assert metrics['snapshots'] == 3
    assert metrics['renders'] == 0


def test_tensor_rank1_static_wave_uses_per_cell_rank1_directions():
    from tensor_rank1_sim import TensorRank1State

    npd.seed_all(19)
    state = TensorRank1State.fixed_capacity(
        active_cells=0,
        height=8,
        width=8,
        active_families=1,
        family_capacity=4,
        cell_capacity=16,
        device='cpu',
    )

    spawned, active_family_count = state.append_static_weighted_wave(1, 8, coeff_scale=0.6)

    alive = state.health > 0
    assert spawned == 8
    assert active_family_count == 2
    assert int(torch.unique(state.family_index[alive]).numel()) == 1
    assert not torch.allclose(state.u_1[alive][0], state.u_1[alive][1])
    assert not torch.allclose(state.v_1[alive][0], state.v_1[alive][1])
    assert not torch.allclose(state.u_2[alive][0], state.u_2[alive][1])
    assert not torch.allclose(state.v_2[alive][0], state.v_2[alive][1])


def test_cli_tensor_engine_help_exposes_render_cadence():
    script = Path(__file__).resolve().parents[1] / 'neural_petri_dish.py'
    result = subprocess.run(
        [sys.executable, str(script), '--help'],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert '--tensor-render-every' in result.stdout
    assert '--metrics-json' in result.stdout


def test_cli_tensor_engine_rejects_block_steps_that_skip_refill_checks():
    script = Path(__file__).resolve().parents[1] / 'neural_petri_dish.py'
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            '--engine',
            npd.SIM_ENGINE_TENSOR_RANK1,
            '--action-device',
            'cpu',
            '--max-frames',
            '1',
            '--tensor-block-steps',
            '30',
            '--no-render',
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 2
    assert '--tensor-block-steps must divide --tensor-static-refill-check-every' in result.stderr


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


def test_tensor_rank1_video_renderer_writes_artifact_and_manifest(tmp_path):
    script = Path(__file__).resolve().parent / 'render_tensor_rank1_video.py'
    output = tmp_path / 'tensor_preview.mp4'
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            '--output',
            str(output),
            '--rounds',
            '1',
            '--render-rounds',
            '1',
            '--round-stride',
            '1',
            '--fps',
            '10',
            '--size',
            '5x5',
            '--initial-cells',
            '4',
            '--seed',
            '7',
            '--action-device',
            'cpu',
            '--cell-size',
            '4',
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
    assert 'rounds_requested: 1' in manifest_text
    assert 'render_rounds: 1' in manifest_text
    assert 'round_stride: 1' in manifest_text
    assert 'frames_written: 17' in manifest_text
    assert 'rendered_rounds: 0' in manifest_text
    assert 'empty_refills: 0' in manifest_text
    assert 'early_ended_rounds: 1' in manifest_text


def test_normal_play_benchmark_exposes_tensor_engine_and_rejects_cpu():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark_normal_play.py'
    help_result = subprocess.run(
        [sys.executable, str(script), '--help'],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert help_result.returncode == 0, help_result.stderr
    assert '--engine {game,tensor_rank1}' in help_result.stdout
    assert '--tensor-family-capacity' in help_result.stdout
    assert '--tensor-cell-capacity' in help_result.stdout

    cpu_result = subprocess.run(
        [
            sys.executable,
            str(script),
            '--engine',
            'tensor_rank1',
            '--action-device',
            'cpu',
            '--rounds',
            '1',
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert cpu_result.returncode == 2
    assert '--engine tensor_rank1 requires --action-device cuda' in cpu_result.stderr


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
            '--action-backend',
            'sequential',
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
        assert 'action_backend: sequential' in manifest_text

    csv_text = (tmp_path / 'survival_rounds_2.csv').read_text(encoding='utf-8')
    assert 'mutation_mode,round,previous_round_cells,pre_refill_cells,post_refill_cells' in csv_text
    assert 'shared_rank1_factored,0,' in csv_text


def test_tensor_rank1_edge_diagnostics_cpu_smoke(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'tensor_rank1_edge_diagnostics.py'
    output_json = tmp_path / 'edge_diagnostics.json'
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            '--device',
            'cpu',
            '--size',
            '8x8',
            '--initial-cells',
            '8',
            '--rounds',
            '1',
            '--coeff-scales',
            '0.001,0.01',
            '--family-capacity',
            '3',
            '--output-json',
            str(output_json),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = json.loads(output_json.read_text(encoding='utf-8'))
    assert [row['coeff_scale'] for row in rows] == [0.001, 0.01]
    assert all(row['steps'] == npd.ROUNDTIME for row in rows)
    assert all('border_hit_rate_per_move' in row for row in rows)
    assert result.stdout.count('"coeff_scale"') == 2
