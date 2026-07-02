import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import neural_petri_dish as npd
from tensor_rank1_sim import (
    DIRECTION_OUTPUT_DIM,
    EVENT_COUNT_NAMES,
    HIDDEN_DIM,
    NEIGHBOR_INPUT_DIM,
    TensorRank1State,
    benchmark_tensor_state,
    family_basis_rebuild_snapshot_combat_block_tensors,
    snapshot_combat_step_tensors,
    snapshot_combat_step_tensors_family_basis_rebuild_grid,
    snapshot_combat_step_tensors_rebuild_grid,
)


def attack_action(direction):
    return DIRECTION_OUTPUT_DIM + direction


def clone_tensor_state(state):
    kwargs = {}
    for field_name in state.__dataclass_fields__:
        value = getattr(state, field_name)
        kwargs[field_name] = value.clone() if isinstance(value, torch.Tensor) else value
    return TensorRank1State(**kwargs)


def assert_tensor_state_position_invariants(state):
    expected_flat = state.positions[:, 0] * state.grid.shape[1] + state.positions[:, 1]
    assert torch.equal(state.flat_positions, expected_flat)
    alive = state.health > 0
    if alive.any():
        assert state.grid.reshape(-1)[state.flat_positions[alive]].eq(1).all()
        live_index_values = state.index_grid.reshape(-1)[state.flat_positions[alive]]
        expected_indices = torch.arange(state.cells, device=state.device, dtype=live_index_values.dtype)[alive]
        assert live_index_values.ge(0).all()
        assert live_index_values.lt(state.cells).all()
        assert torch.equal(live_index_values, expected_indices)
        assert state.health[live_index_values.to(torch.long)].gt(0).all()


def assert_tensor_state_base_matmul_cache(state):
    expected_weight_1 = state.base_weight_1.reshape(
        state.families * state.base_weight_1.shape[1],
        state.base_weight_1.shape[2],
    ).t().contiguous()
    expected_weight_2 = state.base_weight_2.reshape(
        state.families * state.base_weight_2.shape[1],
        state.base_weight_2.shape[2],
    ).t().contiguous()
    assert torch.equal(state.base_weight_1_matmul, expected_weight_1)
    assert torch.equal(state.base_weight_2_matmul, expected_weight_2)


def force_single_cell_action(state, direction, attack=False):
    state.base_weight_1.zero_()
    state.base_weight_2.zero_()
    state.u_1.zero_()
    state.v_1.zero_()
    state.u_2.zero_()
    state.v_2.zero_()
    state.coeff_1.zero_()
    state.coeff_2.zero_()
    state.bias_1.fill_(1)
    state.bias_2.fill_(-10)
    state.bias_2[:, direction] = 10
    state.bias_2[:, npd.ATTACK_OUTPUT_INDEX] = 10 if attack else -10
    state.refresh_base_weight_matmul_cache()


def place_single_cell_and_npc(state, cell_row, cell_col, npc_row, npc_col):
    state.health.zero_()
    state.health[0] = 3
    state.stationary_steps.zero_()
    state.recurrent_state.zero_()
    state.food_grid.zero_()
    state.flat_positions[0] = cell_row * state.grid_stride + cell_col
    state.positions[0, 0] = cell_row
    state.positions[0, 1] = cell_col
    state.family_index[0] = 0
    state.round_survival_steps.zero_()
    state.round_participants.copy_(state.health > 0)
    state.npc_flat_positions = torch.tensor(
        [npc_row * state.grid_stride + npc_col],
        device=state.device,
        dtype=torch.long,
    )
    state.npc_grid.zero_()
    state.npc_grid.reshape(-1)[state.npc_flat_positions] = 1
    state.npc_random_directions = torch.empty(1, 1, device=state.device, dtype=torch.long)
    state.rebuild_grids()


def run_single_family_basis_step(state, npc_direction, collect_event_counts=True):
    state.npc_random_directions[0, 0] = npc_direction
    outputs = snapshot_combat_step_tensors_family_basis_rebuild_grid(
        state.index_grid,
        state.food_grid,
        state.flat_positions,
        state.health,
        state.stationary_steps,
        state.recurrent_state,
        state.round_survival_steps,
        state.npc_flat_positions,
        state.npc_random_directions[0],
        state.family_index,
        state.coeff_1,
        state.coeff_2,
        state.bias_1,
        state.bias_2,
        state.base_weight_1_matmul,
        state.base_weight_2_matmul,
        state.u_1,
        state.v_1,
        state.u_2,
        state.v_2,
        state.stationary_health_cap,
        state.index_grid_indices(),
        state.dead_index_grid_indices(),
        state.neighbor_flat_offsets,
        state.direction_flat_deltas,
        collect_event_counts,
    )
    event_counts = dict(zip(EVENT_COUNT_NAMES, outputs[-1].to(torch.long).tolist()))
    return outputs, event_counts


def test_gpu_mutation_benchmark_cpu_smoke():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'gpu_mutation_benchmark.py'
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            '--device',
            'cpu',
            '--population',
            '32',
            '--steps',
            '3',
            '--warmup-steps',
            '1',
            '--mode',
            'shared_rank1_factored',
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads(result.stdout)
    assert metrics['mode'] == 'shared_rank1_factored'
    assert metrics['device'] == 'cpu'
    assert metrics['representation'] == 'shared_base_rank1_coefficients'
    assert metrics['population'] == 32
    assert metrics['steps'] == 3
    assert metrics['cells_per_second'] > 0


def test_normal_play_benchmark_cpu_smoke():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark_normal_play.py'
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            '--action-backend',
            'sequential',
            '--action-device',
            'cpu',
            '--size',
            '8x8',
            '--initial-cells',
            '10',
            '--rounds',
            '1',
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads(result.stdout)
    assert metrics['action_backend'] == 'sequential'
    assert metrics['action_device'] == 'cpu'
    assert len(metrics['rounds']) == 1
    assert metrics['rounds'][0]['seconds'] > 0


def test_gpu_action_kernel_benchmark_cpu_smoke():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'gpu_action_kernel_benchmark.py'
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            '--device',
            'cpu',
            '--cells',
            '128',
            '--steps',
            '2',
            '--height',
            '16',
            '--width',
            '16',
            '--matmul-precision',
            'high',
            '--compile-mode',
            'default',
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads(result.stdout)
    assert metrics['device'] == 'cpu'
    assert metrics['cells'] == 128
    assert metrics['families'] == 1
    assert metrics['families_final'] == 1
    assert metrics['compact_every'] == 1
    assert metrics['checksum_actions'] == 1024
    assert metrics['compile_mode'] is None
    assert metrics['compiled_block_steps'] is None
    assert metrics['action_checksum'] is not None
    assert metrics['initial_health'] == 2
    assert metrics['matmul_precision'] == 'high'
    assert metrics['movement'] == 'none'
    assert metrics['warmup_steps'] == 20
    assert metrics['cells_per_second'] > 0


def test_static_capacity_sweep_cpu_smoke(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'static_capacity_sweep.py'
    output_json = tmp_path / 'sweep.json'
    output_csv = tmp_path / 'sweep.csv'
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            '--device',
            'cpu',
            '--cells',
            '8',
            '--height',
            '8',
            '--width',
            '8',
            '--steps',
            '2',
            '--warmup-steps',
            '0',
            '--family-capacities',
            '3',
            '--health-dtypes',
            'int32',
            '--matmul-precision',
            'high',
            '--compile-mode',
            'default',
            '--refill-check-everys',
            '1',
            '--no-compiled-step',
            '--no-static-refill-empty',
            '--output-json',
            str(output_json),
            '--output-csv',
            str(output_csv),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    rows = json.loads(output_json.read_text(encoding='utf-8'))
    assert len(rows) == 1
    assert rows[0]['static_capacity'] is True
    assert rows[0]['health_dtype'] == 'int32'
    assert rows[0]['matmul_precision'] == 'high'
    assert rows[0]['compile_mode'] is None
    assert rows[0]['compiled_block_steps'] is None
    assert rows[0]['warmup_steps'] == 0
    assert rows[0]['family_capacity'] == 3
    assert output_csv.exists()


def test_gpu_action_kernel_profile_help_smoke():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'profile_gpu_action_kernel.py'
    result = subprocess.run(
        [sys.executable, str(script), '--help'],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert '--compiled-block-steps' in result.stdout
    assert '--profile-blocks' in result.stdout


def test_tensor_normal_rounds_help_smoke():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark_tensor_normal_rounds.py'
    result = subprocess.run(
        [sys.executable, str(script), '--help'],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert '--rounds' in result.stdout
    assert '--roundtime' in result.stdout
    assert '--cell-capacity' in result.stdout
    assert '--no-cuda-graph-block' in result.stdout


def test_gpu_action_kernel_benchmark_snapshot_movement_cpu_smoke():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'gpu_action_kernel_benchmark.py'
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            '--device',
            'cpu',
            '--cells',
            '128',
            '--steps',
            '2',
            '--height',
            '16',
            '--width',
            '16',
            '--movement',
            'snapshot',
            '--families',
            '4',
            '--initial-health',
            '5',
            '--wave-every',
            '1',
            '--wave-size',
            '4',
            '--compact-every',
            '2',
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads(result.stdout)
    assert metrics['device'] == 'cpu'
    assert metrics['cells'] == 128
    assert metrics['families'] == 4
    assert metrics['families_final'] >= 5
    assert metrics['initial_health'] == 5
    assert metrics['movement'] == 'snapshot'
    assert metrics['wave_every'] == 1
    assert metrics['wave_size'] == 4
    assert metrics['compact_every'] == 2
    assert metrics['waves_spawned'] > 0
    assert metrics['cells_per_second'] > 0


def test_tensor_rank1_state_steps_on_cpu():
    torch.manual_seed(123)
    state = TensorRank1State.random(
        cells=64,
        height=16,
        width=16,
        families=4,
        device=torch.device('cpu'),
    )

    actions = state.step(movement='snapshot')

    assert actions.shape == (64,)
    assert state.positions.shape == (64, 2)
    assert state.recurrent_state.shape == (64, HIDDEN_DIM)
    assert state.grid[state.positions[:, 0], state.positions[:, 1]].eq(1).all()
    assert_tensor_state_position_invariants(state)


def test_tensor_rank1_grid_is_compact_integer_but_inputs_are_float_cpu():
    torch.manual_seed(123)
    state = TensorRank1State.random(
        cells=8,
        height=10,
        width=10,
        families=1,
        device=torch.device('cpu'),
    )

    inputs = state.gather_inputs()

    assert state.grid.dtype == torch.int8
    assert state.index_grid.dtype == torch.int32
    assert inputs.dtype == torch.float32
    assert inputs[:, :NEIGHBOR_INPUT_DIM].min() >= -1
    assert inputs[:, :NEIGHBOR_INPUT_DIM].max() <= npd.FOOD_INPUT_VALUE


def test_tensor_rank1_state_snapshot_combat_compacts_on_cpu():
    torch.manual_seed(123)
    state = TensorRank1State.random(
        cells=64,
        height=16,
        width=16,
        families=4,
        device=torch.device('cpu'),
    )
    state.health.fill_(1)

    state.step(movement='snapshot_combat')

    assert state.positions.shape[0] == state.health.shape[0]
    assert state.recurrent_state.shape[0] == state.health.shape[0]
    assert state.family_index.shape[0] == state.health.shape[0]
    assert state.coeff_1.shape[0] == state.health.shape[0]
    assert state.bias_1.shape[0] == state.health.shape[0]
    assert state.grid[state.positions[:, 0], state.positions[:, 1]].eq(1).all()
    assert_tensor_state_position_invariants(state)


def test_snapshot_combat_tensor_step_matches_eager_cpu():
    torch.manual_seed(123)
    eager = TensorRank1State.random(
        cells=32,
        height=12,
        width=12,
        families=3,
        device=torch.device('cpu'),
        initial_health=5,
    )
    tensor_step = clone_tensor_state(eager)

    eager_actions = eager.step(movement='snapshot_combat', compact_dead=False, sync_positions=False)
    flat_positions, health, recurrent_state, tensor_actions = snapshot_combat_step_tensors(
        tensor_step.grid,
        tensor_step.index_grid,
        tensor_step.food_grid,
        tensor_step.flat_positions,
        tensor_step.health,
        tensor_step.recurrent_state,
        tensor_step.family_index,
        tensor_step.coeff_1,
        tensor_step.coeff_2,
        tensor_step.bias_1,
        tensor_step.bias_2,
        tensor_step.base_weight_1,
        tensor_step.base_weight_2,
        tensor_step.u_1,
        tensor_step.v_1,
        tensor_step.u_2,
        tensor_step.v_2,
        tensor_step.neighbor_flat_offsets,
        tensor_step.direction_flat_deltas,
    )

    assert torch.equal(tensor_actions, eager_actions)
    assert torch.equal(flat_positions, eager.flat_positions)
    assert torch.equal(health, eager.health)
    assert torch.allclose(recurrent_state, eager.recurrent_state, atol=1e-6)
    assert torch.equal(tensor_step.grid, eager.grid)
    assert torch.equal(tensor_step.index_grid, eager.index_grid)
    assert torch.equal(tensor_step.food_grid, eager.food_grid)


def test_snapshot_combat_rebuild_grid_step_matches_eager_cpu():
    torch.manual_seed(123)
    eager = TensorRank1State.random(
        cells=32,
        height=12,
        width=12,
        families=3,
        device=torch.device('cpu'),
        initial_health=5,
    )
    tensor_step = clone_tensor_state(eager)

    eager_actions = eager.step(movement='snapshot_combat', compact_dead=False, sync_positions=False)
    flat_positions, health, recurrent_state, tensor_actions = snapshot_combat_step_tensors_rebuild_grid(
        tensor_step.grid,
        tensor_step.index_grid,
        tensor_step.food_grid,
        tensor_step.flat_positions,
        tensor_step.health,
        tensor_step.recurrent_state,
        tensor_step.family_index,
        tensor_step.coeff_1,
        tensor_step.coeff_2,
        tensor_step.bias_1,
        tensor_step.bias_2,
        tensor_step.base_weight_1,
        tensor_step.base_weight_2,
        tensor_step.u_1,
        tensor_step.v_1,
        tensor_step.u_2,
        tensor_step.v_2,
        tensor_step.neighbor_flat_offsets,
        tensor_step.direction_flat_deltas,
    )

    assert torch.equal(tensor_actions, eager_actions)
    assert torch.equal(flat_positions, eager.flat_positions)
    assert torch.equal(health, eager.health)
    assert torch.allclose(recurrent_state, eager.recurrent_state, atol=1e-6)
    assert torch.equal(tensor_step.grid, eager.grid)
    assert torch.equal(tensor_step.index_grid, eager.index_grid)
    assert torch.equal(tensor_step.food_grid, eager.food_grid)


def test_snapshot_combat_family_basis_rebuild_step_matches_eager_cpu():
    torch.manual_seed(123)
    eager = TensorRank1State.random(
        cells=32,
        height=12,
        width=12,
        families=3,
        device=torch.device('cpu'),
        initial_health=5,
        npc_count=0,
    )
    tensor_step = clone_tensor_state(eager)

    eager_actions = eager.step(movement='snapshot_combat', compact_dead=False, sync_positions=False)
    outputs = snapshot_combat_step_tensors_family_basis_rebuild_grid(*tensor_step.family_basis_step_args())
    flat_positions, health, stationary_steps, recurrent_state, tensor_actions = outputs[:5]

    assert torch.equal(tensor_actions, eager_actions)
    assert torch.equal(flat_positions, eager.flat_positions)
    assert torch.equal(health, eager.health)
    assert torch.equal(stationary_steps, eager.stationary_steps)
    assert torch.allclose(recurrent_state, eager.recurrent_state, atol=1e-6)
    assert torch.equal(tensor_step.index_grid, eager.index_grid)
    assert torch.equal(tensor_step.food_grid, eager.food_grid)


def test_snapshot_combat_family_basis_rebuild_step_matches_eager_for_multiple_steps_cpu():
    torch.manual_seed(123)
    eager = TensorRank1State.random(
        cells=32,
        height=12,
        width=12,
        families=3,
        device=torch.device('cpu'),
        initial_health=5,
        npc_count=0,
    )
    tensor_step = clone_tensor_state(eager)

    for _ in range(2):
        eager_actions = eager.step(movement='snapshot_combat', compact_dead=False, sync_positions=False)
        outputs = snapshot_combat_step_tensors_family_basis_rebuild_grid(*tensor_step.family_basis_step_args())
        flat_positions, health, stationary_steps, recurrent_state, tensor_actions = outputs[:5]
        tensor_step.flat_positions = flat_positions
        tensor_step.health = health
        tensor_step.stationary_steps = stationary_steps
        tensor_step.recurrent_state = recurrent_state
        tensor_step.round_survival_steps = outputs[5]
        tensor_step.npc_flat_positions = outputs[6]

        assert torch.equal(tensor_actions, eager_actions)
        assert torch.equal(tensor_step.flat_positions, eager.flat_positions)
        assert torch.equal(tensor_step.health, eager.health)
        assert torch.equal(tensor_step.stationary_steps, eager.stationary_steps)
        assert torch.allclose(tensor_step.recurrent_state, eager.recurrent_state, atol=1e-6)
        assert torch.equal(tensor_step.index_grid, eager.index_grid)
        assert torch.equal(tensor_step.food_grid, eager.food_grid)


def test_snapshot_combat_family_basis_rebuild_block_matches_eager_cpu():
    torch.manual_seed(123)
    eager = TensorRank1State.random(
        cells=32,
        height=12,
        width=12,
        families=3,
        device=torch.device('cpu'),
        initial_health=5,
        npc_count=0,
    )
    tensor_step = clone_tensor_state(eager)
    block_fn = family_basis_rebuild_snapshot_combat_block_tensors(2)

    for _ in range(2):
        eager.step(movement='snapshot_combat', compact_dead=False, sync_positions=False)
    tensor_step.refresh_npc_random_directions(2)
    outputs = block_fn(*tensor_step.family_basis_block_args())
    flat_positions, health, stationary_steps, recurrent_state = outputs[:4]

    assert torch.equal(flat_positions, eager.flat_positions)
    assert torch.equal(health, eager.health)
    assert torch.equal(stationary_steps, eager.stationary_steps)
    assert torch.allclose(recurrent_state, eager.recurrent_state, atol=1e-6)
    assert torch.equal(tensor_step.index_grid, eager.index_grid)
    assert torch.equal(tensor_step.food_grid, eager.food_grid)


def test_snapshot_combat_stationary_action_damages_and_can_kill_cpu():
    state = TensorRank1State.random(
        cells=4,
        height=8,
        width=8,
        families=1,
        device=torch.device('cpu'),
        initial_health=2,
        stationary_health_cap=1,
        npc_count=0,
    )
    state.base_weight_1.zero_()
    state.base_weight_2.zero_()
    state.refresh_base_weight_matmul_cache()
    state.u_1.zero_()
    state.v_1.zero_()
    state.u_2.zero_()
    state.v_2.zero_()
    state.coeff_1.zero_()
    state.coeff_2.zero_()
    state.bias_1.zero_()
    state.bias_2.fill_(-100)
    state.bias_2[:, 0] = 100

    health = state.health
    stationary_steps = state.stationary_steps
    flat_positions = state.flat_positions
    recurrent_state = state.recurrent_state
    round_survival_steps = state.round_survival_steps
    state.refresh_npc_random_directions(1)
    for _ in range(4):
        outputs = snapshot_combat_step_tensors_family_basis_rebuild_grid(
            state.index_grid,
            state.food_grid,
            flat_positions,
            health,
            stationary_steps,
            recurrent_state,
            round_survival_steps,
            state.npc_flat_positions,
            state.npc_random_directions[0],
            state.family_index,
            state.coeff_1,
            state.coeff_2,
            state.bias_1,
            state.bias_2,
            state.base_weight_1_matmul,
            state.base_weight_2_matmul,
            state.u_1,
            state.v_1,
            state.u_2,
            state.v_2,
            state.stationary_health_cap,
            state.index_grid_indices(),
            state.dead_index_grid_indices(),
            state.neighbor_flat_offsets,
            state.direction_flat_deltas,
        )
        flat_positions, health, stationary_steps, recurrent_state, actions, round_survival_steps = outputs[:6]

    assert torch.equal(actions, torch.zeros_like(actions))
    assert torch.equal(flat_positions, state.flat_positions)
    assert torch.equal(health, torch.zeros_like(health))
    assert torch.equal(stationary_steps, torch.zeros_like(stationary_steps))
    assert state.index_grid.reshape(-1)[flat_positions].eq(-1).all()


def test_snapshot_combat_family_basis_sanitizes_nan_recurrent_state_cpu():
    state = TensorRank1State.random(
        cells=8,
        height=8,
        width=8,
        families=2,
        device=torch.device('cpu'),
        initial_health=5,
        npc_count=0,
    )
    state.recurrent_state.fill_(float('nan'))

    outputs = snapshot_combat_step_tensors_family_basis_rebuild_grid(*state.family_basis_step_args())
    _flat_positions, _health, _stationary_steps, recurrent_state, actions = outputs[:5]

    assert torch.isfinite(recurrent_state).all()
    assert actions.ge(0).all()
    assert actions.lt(DIRECTION_OUTPUT_DIM * 2).all()


def test_snapshot_combat_kill_moves_attacker_and_rewards_health_cpu():
    state = TensorRank1State.random(
        cells=2,
        height=8,
        width=8,
        families=1,
        device=torch.device('cpu'),
        initial_health=2,
    )
    state.positions = torch.tensor([[3, 3], [3, 4]], dtype=torch.long)
    state.flat_positions = state.positions[:, 0] * state.grid_stride + state.positions[:, 1]
    state.health = torch.tensor([2, 1], dtype=state.health.dtype)
    state.stationary_steps.zero_()
    state.rebuild_grids()

    state.apply_snapshot_combat(
        torch.tensor([attack_action(3), 0], dtype=torch.long),
        compact_dead=False,
        sync_positions=False,
    )

    assert state.health.tolist() == [7, 0]
    assert int(state.flat_positions[0].item()) == 3 * state.grid_stride + 4
    assert int(state.index_grid.reshape(-1)[state.flat_positions[0]].item()) == 0


def test_snapshot_combat_lone_target_takes_extra_damage_cpu():
    state = TensorRank1State.random(
        cells=2,
        height=8,
        width=8,
        families=1,
        device=torch.device('cpu'),
        initial_health=2,
    )
    state.positions = torch.tensor([[3, 3], [3, 4]], dtype=torch.long)
    state.flat_positions = state.positions[:, 0] * state.grid_stride + state.positions[:, 1]
    state.health = torch.tensor([2, 2], dtype=state.health.dtype)
    state.stationary_steps.zero_()
    state.rebuild_grids()

    state.apply_snapshot_combat(
        torch.tensor([attack_action(3), 0], dtype=torch.long),
        compact_dead=False,
        sync_positions=False,
    )

    assert state.health.tolist() == [7, 0]
    assert int(state.flat_positions[0].item()) == 3 * state.grid_stride + 4


def test_snapshot_combat_protected_target_avoids_extra_damage_cpu():
    state = TensorRank1State.random(
        cells=3,
        height=8,
        width=8,
        families=1,
        device=torch.device('cpu'),
        initial_health=2,
    )
    state.positions = torch.tensor([[3, 3], [3, 4], [4, 4]], dtype=torch.long)
    state.flat_positions = state.positions[:, 0] * state.grid_stride + state.positions[:, 1]
    state.health = torch.tensor([3, 2, 2], dtype=state.health.dtype)
    state.stationary_steps.zero_()
    state.rebuild_grids()

    state.apply_snapshot_combat(
        torch.tensor([attack_action(3), 0, 0], dtype=torch.long),
        compact_dead=False,
        sync_positions=False,
    )

    assert state.health.tolist() == [2, 1, 2]
    assert int(state.flat_positions[0].item()) == 3 * state.grid_stride + 3
    assert_tensor_state_position_invariants(state)


def test_snapshot_combat_move_intent_into_occupied_square_misses_cpu():
    state = TensorRank1State.random(
        cells=2,
        height=8,
        width=8,
        families=1,
        device=torch.device('cpu'),
        initial_health=2,
    )
    state.positions = torch.tensor([[3, 3], [3, 4]], dtype=torch.long)
    state.flat_positions = state.positions[:, 0] * state.grid_stride + state.positions[:, 1]
    state.health = torch.tensor([2, 2], dtype=state.health.dtype)
    state.stationary_steps.zero_()
    state.rebuild_grids()

    state.apply_snapshot_combat(
        torch.tensor([3, 0], dtype=torch.long),
        compact_dead=False,
        sync_positions=False,
    )

    assert state.health.tolist() == [2, 2]
    assert int(state.flat_positions[0].item()) == 3 * state.grid_stride + 3
    assert int(state.flat_positions[1].item()) == 3 * state.grid_stride + 4
    assert_tensor_state_position_invariants(state)


def test_snapshot_combat_attack_intent_into_empty_square_misses_cpu():
    state = TensorRank1State.random(
        cells=1,
        height=8,
        width=8,
        families=1,
        device=torch.device('cpu'),
        initial_health=2,
    )
    state.positions = torch.tensor([[3, 3]], dtype=torch.long)
    state.flat_positions = state.positions[:, 0] * state.grid_stride + state.positions[:, 1]
    state.health = torch.tensor([2], dtype=state.health.dtype)
    state.stationary_steps.zero_()
    state.rebuild_grids()

    state.apply_snapshot_combat(
        torch.tensor([attack_action(3)], dtype=torch.long),
        compact_dead=False,
        sync_positions=False,
    )

    assert state.health.tolist() == [2]
    assert int(state.flat_positions[0].item()) == 3 * state.grid_stride + 3
    assert_tensor_state_position_invariants(state)


def test_snapshot_combat_successful_move_costs_fractional_health_cpu():
    state = TensorRank1State.random(
        cells=1,
        height=8,
        width=8,
        families=1,
        device=torch.device('cpu'),
        initial_health=2,
    )
    state.positions = torch.tensor([[3, 3]], dtype=torch.long)
    state.flat_positions = state.positions[:, 0] * state.grid_stride + state.positions[:, 1]
    state.health = torch.tensor([2.0], dtype=state.health.dtype)
    state.stationary_steps.zero_()
    state.food_grid.zero_()
    state.rebuild_grids()

    state.apply_snapshot_combat(
        torch.tensor([3], dtype=torch.long),
        compact_dead=False,
        sync_positions=True,
    )

    assert torch.allclose(state.health, torch.tensor([1.9], dtype=state.health.dtype))
    assert int(state.flat_positions[0].item()) == 3 * state.grid_stride + 4
    assert_tensor_state_position_invariants(state)


def test_snapshot_combat_successful_move_can_kill_without_health_floor_cpu():
    state = TensorRank1State.random(
        cells=1,
        height=8,
        width=8,
        families=1,
        device=torch.device('cpu'),
        initial_health=2,
    )
    state.positions = torch.tensor([[3, 3]], dtype=torch.long)
    state.flat_positions = state.positions[:, 0] * state.grid_stride + state.positions[:, 1]
    state.health = torch.tensor([0.05], dtype=state.health.dtype)
    state.stationary_steps.zero_()
    state.rebuild_grids()

    state.apply_snapshot_combat(
        torch.tensor([3], dtype=torch.long),
        compact_dead=False,
        sync_positions=True,
    )

    assert state.health.tolist() == [0.0]
    assert int(state.index_grid.reshape(-1)[3 * state.grid_stride + 4].item()) == -1
    assert_tensor_state_position_invariants(state)


def test_snapshot_combat_collision_losers_do_not_stay_alive_cpu():
    state = TensorRank1State.random(
        cells=3,
        height=8,
        width=8,
        families=1,
        device=torch.device('cpu'),
        initial_health=2,
    )
    state.positions = torch.tensor([[3, 3], [3, 5], [3, 4]], dtype=torch.long)
    state.flat_positions = state.positions[:, 0] * state.grid_stride + state.positions[:, 1]
    state.health = torch.tensor([2, 2, 1], dtype=state.health.dtype)
    state.stationary_steps.zero_()
    state.rebuild_grids()

    state.apply_snapshot_combat(
        torch.tensor([attack_action(3), attack_action(4), 0], dtype=torch.long),
        compact_dead=False,
        sync_positions=False,
    )

    assert int((state.health > 0).sum().item()) == 1
    assert int(state.index_grid.reshape(-1)[3 * state.grid_stride + 4].item()) in (0, 1)
    state.sync_positions_from_flat()
    assert_tensor_state_position_invariants(state)


def test_tensor_rank1_state_deferred_compaction_removes_dead_cells_from_grid():
    torch.manual_seed(123)
    state = TensorRank1State.random(
        cells=64,
        height=16,
        width=16,
        families=4,
        device=torch.device('cpu'),
    )
    state.health.fill_(1)

    state.step(movement='snapshot_combat', compact_dead=False)

    alive = state.health > 0
    assert state.positions.shape[0] == 64
    if alive.any():
        assert state.grid[state.positions[alive, 0], state.positions[alive, 1]].eq(1).all()
    if (~alive).any():
        dead_positions = state.positions[~alive]
        dead_indices = torch.arange(64)[~alive]
        assert state.index_grid[dead_positions[:, 0], dead_positions[:, 1]].ne(dead_indices).all()

    state.compact(alive)
    assert state.positions.shape[0] == int(alive.sum())
    assert_tensor_state_position_invariants(state)


def test_tensor_rank1_compaction_prunes_unused_families_without_changing_survivor_weights():
    torch.manual_seed(123)
    state = TensorRank1State.random(
        cells=12,
        height=12,
        width=12,
        families=4,
        device=torch.device('cpu'),
    )
    state.family_index = torch.tensor([0, 1, 2, 3, 1, 2, 3, 0, 2, 3, 1, 0], dtype=torch.long)
    alive = state.family_index == 2
    expected_weight_1 = state.dense_weight_1()[alive].clone()
    expected_weight_2 = state.dense_weight_2()[alive].clone()

    state.compact(alive)

    assert state.families == 1
    assert state.family_index.eq(0).all()
    assert state.single_active_family_id == 0
    assert torch.allclose(state.dense_weight_1(), expected_weight_1)
    assert torch.allclose(state.dense_weight_2(), expected_weight_2)
    assert_tensor_state_position_invariants(state)


def test_tensor_rank1_state_weighted_wave_uses_standardized_survival_update():
    torch.manual_seed(123)
    state = TensorRank1State.random(
        cells=8,
        height=10,
        width=10,
        families=3,
        device=torch.device('cpu'),
    )
    old_cells = state.cells
    old_families = state.families
    state.round_survival_steps = torch.linspace(0, 500, old_cells)
    state.round_participants = torch.ones(old_cells, dtype=torch.bool)
    fitness = state.round_survival_steps / 500.0
    advantage = (fitness - fitness.mean()) / fitness.std(unbiased=False)
    expected_weight_1 = state.evolution_anchor_weight_1 + npd.FITNESS_UPDATE_LR * (
        advantage.reshape(-1, 1, 1)
        * (state.dense_weight_1() - state.evolution_anchor_weight_1.unsqueeze(0))
    ).mean(dim=0)
    expected_weight_2 = state.evolution_anchor_weight_2 + npd.FITNESS_UPDATE_LR * (
        advantage.reshape(-1, 1, 1)
        * (state.dense_weight_2() - state.evolution_anchor_weight_2.unsqueeze(0))
    ).mean(dim=0)
    expected_bias_1 = state.evolution_anchor_bias_1 + npd.FITNESS_UPDATE_LR * (
        advantage.reshape(-1, 1)
        * (state.bias_1 - state.evolution_anchor_bias_1.unsqueeze(0))
    ).mean(dim=0)
    old_family_index = state.family_index.clone()

    spawned = state.append_weighted_wave(5, initial_health=7)

    assert spawned == 5
    assert state.cells == old_cells + 5
    assert state.families == old_families + 1
    assert torch.equal(state.family_index[:old_cells], old_family_index)
    assert state.family_index[old_cells:].eq(old_families).all()
    assert state.health[old_cells:].eq(7).all()
    assert torch.allclose(state.base_weight_1[-1], expected_weight_1, atol=1e-6)
    assert torch.allclose(state.base_weight_2[-1], expected_weight_2, atol=1e-6)
    assert torch.allclose(state.bias_1[old_cells], expected_bias_1, atol=1e-6)
    assert_tensor_state_base_matmul_cache(state)
    assert_tensor_state_position_invariants(state)


def test_tensor_rank1_round_penalty_applies_after_family_update_cpu():
    torch.manual_seed(123)
    state = TensorRank1State.fixed_capacity(
        active_cells=4,
        height=10,
        width=10,
        active_families=1,
        family_capacity=3,
        device=torch.device('cpu'),
        initial_health=1,
        cell_capacity=8,
        npc_count=0,
    )
    state.round_survival_steps = torch.linspace(0, npd.ROUNDTIME, state.cells)
    state.round_participants = torch.ones(state.cells, dtype=torch.bool)

    pre_penalty_family = state.weighted_survivor_family()
    state.apply_round_transition_health_cost()
    spawned, family_count = state.append_static_weighted_wave(
        1,
        4,
        initial_health=7,
        precomputed_family=pre_penalty_family,
    )

    assert spawned == 4
    assert family_count == 2
    assert int((state.health == 7).sum().item()) == 4
    assert torch.equal(state.base_weight_1[1], pre_penalty_family[0])
    assert torch.equal(state.base_weight_2[1], pre_penalty_family[2])
    assert torch.equal(state.bias_1[state.health == 7][0], pre_penalty_family[1])
    assert torch.equal(state.bias_2[state.health == 7][0], pre_penalty_family[3])


def test_tensor_rank1_fixed_capacity_inactive_family_slots_do_not_affect_rng_cpu():
    kwargs = dict(
        active_cells=4,
        height=10,
        width=10,
        active_families=2,
        device=torch.device('cpu'),
        initial_health=3,
        cell_capacity=10,
    )
    torch.manual_seed(123)
    tight = TensorRank1State.fixed_capacity(family_capacity=2, **kwargs)
    next_after_tight = torch.rand(4)

    torch.manual_seed(123)
    padded = TensorRank1State.fixed_capacity(family_capacity=5, **kwargs)
    next_after_padded = torch.rand(4)

    for field_name in (
            'positions',
            'flat_positions',
            'health',
            'recurrent_state',
            'family_index',
            'coeff_1',
            'coeff_2',
            'bias_1',
            'bias_2'):
        assert torch.equal(getattr(tight, field_name), getattr(padded, field_name))
    assert torch.equal(tight.base_weight_1, padded.base_weight_1[:2])
    assert torch.equal(tight.base_weight_2, padded.base_weight_2[:2])
    assert torch.equal(tight.u_1, padded.u_1)
    assert torch.equal(tight.v_1, padded.v_1)
    assert torch.equal(tight.u_2, padded.u_2)
    assert torch.equal(tight.v_2, padded.v_2)
    assert padded.base_weight_1[2:].eq(0).all()
    assert padded.base_weight_2[2:].eq(0).all()
    assert torch.equal(next_after_tight, next_after_padded)
    assert_tensor_state_base_matmul_cache(padded)


def test_tensor_rank1_fixed_capacity_static_wave_keeps_shapes_cpu():
    torch.manual_seed(123)
    state = TensorRank1State.fixed_capacity(
        active_cells=8,
        height=10,
        width=10,
        active_families=1,
        family_capacity=4,
        device=torch.device('cpu'),
        initial_health=3,
        cell_capacity=20,
    )

    old_cells = state.cells
    old_family_capacity = state.families
    spawned, family_count = state.append_static_weighted_wave(1, 5, initial_health=7)

    assert spawned == 5
    assert family_count == 2
    assert state.cells == old_cells
    assert state.cells == 20
    assert state.families == old_family_capacity
    assert int((state.health > 0).sum()) == 13
    assert state.family_index[state.health > 0].max().item() == 1
    assert_tensor_state_base_matmul_cache(state)
    assert_tensor_state_position_invariants(state)


def test_tensor_rank1_network_dtype_only_changes_network_tensors_cpu():
    torch.manual_seed(123)
    state = TensorRank1State.fixed_capacity(
        active_cells=4,
        height=10,
        width=10,
        active_families=1,
        family_capacity=3,
        device=torch.device('cpu'),
        initial_health=3,
        cell_capacity=10,
        network_dtype=torch.float16,
    )

    assert state.health.dtype == torch.float32
    assert state.stationary_steps.dtype == torch.int16
    assert state.round_survival_steps.dtype == torch.float32
    for tensor in (
            state.recurrent_state,
            state.coeff_1,
            state.coeff_2,
            state.bias_1,
            state.bias_2,
            state.base_weight_1,
            state.base_weight_2,
            state.base_weight_1_matmul,
            state.base_weight_2_matmul,
            state.u_1,
            state.v_1,
            state.u_2,
            state.v_2):
        assert tensor.dtype == torch.float16


def test_tensor_rank1_static_wave_reuses_dead_family_slot_cpu():
    torch.manual_seed(123)
    state = TensorRank1State.fixed_capacity(
        active_cells=4,
        height=10,
        width=10,
        active_families=2,
        family_capacity=2,
        device=torch.device('cpu'),
        initial_health=3,
        cell_capacity=10,
    )
    state.family_index[:4] = torch.tensor([0, 0, 1, 1], device=state.device)
    state.health[:2] = 0
    survivor_family_weight = state.base_weight_1[1].clone()

    spawned, family_count = state.append_static_weighted_wave(2, 3, initial_health=7)

    assert spawned == 3
    assert family_count == 2
    assert state.families == 2
    assert state.live_family_count() == 2
    assert torch.equal(state.family_index[state.health == 7], torch.zeros(3, dtype=state.family_index.dtype))
    assert torch.allclose(state.base_weight_1[1], survivor_family_weight)
    assert_tensor_state_base_matmul_cache(state)
    assert_tensor_state_position_invariants(state)


def test_tensor_rank1_static_wave_grows_when_all_family_slots_are_live_cpu():
    torch.manual_seed(123)
    state = TensorRank1State.fixed_capacity(
        active_cells=4,
        height=10,
        width=10,
        active_families=2,
        family_capacity=2,
        device=torch.device('cpu'),
        initial_health=3,
        cell_capacity=10,
    )
    state.family_index[:4] = torch.tensor([0, 0, 1, 1], device=state.device)
    version = state.family_capacity_version()

    spawned, family_count = state.append_static_weighted_wave(2, 3, initial_health=7)

    assert spawned == 3
    assert family_count == 3
    assert state.families == 4
    assert state.family_capacity_version() == version + 1
    assert state.live_family_count() == 3
    assert state.family_index[state.health == 7].eq(2).all()
    assert_tensor_state_base_matmul_cache(state)
    assert_tensor_state_position_invariants(state)


def test_tensor_rank1_cell_dies_when_moving_into_npc_old_tile_cpu():
    state = TensorRank1State.fixed_capacity(
        active_cells=1,
        height=8,
        width=8,
        active_families=1,
        family_capacity=1,
        device=torch.device('cpu'),
        initial_health=3,
        cell_capacity=1,
        npc_count=0,
    )
    place_single_cell_and_npc(state, cell_row=4, cell_col=4, npc_row=4, npc_col=5)
    force_single_cell_action(state, direction=3, attack=False)

    outputs, event_counts = run_single_family_basis_step(state, npc_direction=3)
    new_flat_positions, new_health, *_rest = outputs

    assert int(new_flat_positions[0].item()) == 4 * state.grid_stride + 5
    assert float(new_health[0].item()) == 0.0
    assert event_counts['npc_kills'] == 1
    assert event_counts['npc_visible_cell_steps'] == 1
    assert event_counts['npc_visible_deaths'] == 1
    assert event_counts['npc_visible_npc_kills'] == 1
    assert event_counts['npc_visible_final_alive'] == 0
    assert event_counts['npc_visible_final_clear'] == 0


def test_tensor_rank1_cell_dies_when_npc_moves_into_cell_cpu():
    state = TensorRank1State.fixed_capacity(
        active_cells=1,
        height=8,
        width=8,
        active_families=1,
        family_capacity=1,
        device=torch.device('cpu'),
        initial_health=3,
        cell_capacity=1,
        npc_count=0,
    )
    place_single_cell_and_npc(state, cell_row=4, cell_col=4, npc_row=4, npc_col=5)
    force_single_cell_action(state, direction=0, attack=False)

    outputs, event_counts = run_single_family_basis_step(state, npc_direction=4)
    _new_flat_positions, new_health, _stationary_steps, _hidden, _actions, _survival, new_npc_positions, _counts = outputs

    assert int(new_npc_positions[0].item()) == 4 * state.grid_stride + 4
    assert float(new_health[0].item()) == 0.0
    assert event_counts['npc_kills'] == 1
    assert event_counts['stayed_put'] == 1
    assert event_counts['npc_visible_cell_steps'] == 1
    assert event_counts['npc_visible_deaths'] == 1
    assert event_counts['npc_visible_npc_kills'] == 1
    assert event_counts['npc_visible_final_alive'] == 0
    assert event_counts['npc_visible_final_clear'] == 0


def test_tensor_rank1_npc_visible_escape_metrics_use_final_npc_position_cpu():
    state = TensorRank1State.fixed_capacity(
        active_cells=1,
        height=8,
        width=8,
        active_families=1,
        family_capacity=1,
        device=torch.device('cpu'),
        initial_health=3,
        cell_capacity=1,
        npc_count=0,
    )
    place_single_cell_and_npc(state, cell_row=4, cell_col=4, npc_row=4, npc_col=5)
    force_single_cell_action(state, direction=4, attack=False)

    outputs, event_counts = run_single_family_basis_step(state, npc_direction=3)
    new_flat_positions, new_health, *_rest = outputs

    assert int(new_flat_positions[0].item()) == 4 * state.grid_stride + 3
    assert float(new_health[0].item()) == pytest.approx(3 - npd.MOVEMENT_HEALTH_COST)
    assert event_counts['npc_visible_cell_steps'] == 1
    assert event_counts['npc_visible_move_away'] == 1
    assert event_counts['npc_visible_deaths'] == 0
    assert event_counts['npc_visible_final_alive'] == 1
    assert event_counts['npc_visible_final_clear'] == 1
    assert event_counts['npc_visible_final_adjacent'] == 0
    assert event_counts['npc_visible_final_farther'] == 1
    assert event_counts['npc_visible_final_closer'] == 0


def test_tensor_rank1_disabling_event_counts_preserves_step_outputs_cpu():
    state = TensorRank1State.fixed_capacity(
        active_cells=1,
        height=8,
        width=8,
        active_families=1,
        family_capacity=1,
        device=torch.device('cpu'),
        initial_health=3,
        cell_capacity=1,
        npc_count=0,
    )
    place_single_cell_and_npc(state, cell_row=4, cell_col=4, npc_row=4, npc_col=5)
    force_single_cell_action(state, direction=4, attack=False)
    metrics_state = clone_tensor_state(state)
    fast_state = clone_tensor_state(state)

    metrics_outputs, metrics_counts = run_single_family_basis_step(metrics_state, npc_direction=3, collect_event_counts=True)
    fast_outputs, fast_counts = run_single_family_basis_step(fast_state, npc_direction=3, collect_event_counts=False)

    for metrics_value, fast_value in zip(metrics_outputs[:-1], fast_outputs[:-1]):
        assert torch.equal(metrics_value, fast_value)
    assert metrics_counts['npc_visible_final_clear'] == 1
    assert all(value == 0 for value in fast_counts.values())


def test_tensor_rank1_round_transition_health_cost_can_kill_survivors_cpu():
    torch.manual_seed(123)
    state = TensorRank1State.random(
        cells=3,
        height=8,
        width=8,
        families=1,
        device=torch.device('cpu'),
        initial_health=3,
    )
    state.health = torch.tensor([1, 3, 4], dtype=state.health.dtype)
    state.rebuild_grids()

    state.apply_round_transition_health_cost()

    assert state.health.tolist() == [0, 0.5, 1.5]
    assert int((state.health > 0).sum().item()) == 2
    assert int(state.index_grid.reshape(-1)[state.flat_positions[0]].item()) == -1
    assert int(state.index_grid.reshape(-1)[state.flat_positions[1]].item()) == 1
    assert_tensor_state_position_invariants(state)


def test_tensor_rank1_flat_empty_positions_match_position_view():
    torch.manual_seed(123)
    state = TensorRank1State.random(
        cells=8,
        height=10,
        width=10,
        families=3,
        device=torch.device('cpu'),
    )

    empty_positions = state.empty_positions()
    empty_flat = state.empty_flat_positions()

    assert torch.equal(empty_flat, empty_positions[:, 0] * state.grid.shape[1] + empty_positions[:, 1])


def test_tensor_rank1_fixed_food_spawns_in_same_positions_cpu():
    torch.manual_seed(123)
    state = TensorRank1State.random(
        cells=4,
        height=12,
        width=12,
        families=1,
        device=torch.device('cpu'),
    )
    first_food = state.food_grid.clone()
    first_positions = state.food_flat_positions.clone()

    state.food_grid.zero_()
    state.spawn_fixed_food()

    assert int(state.food_grid.sum().item()) == min(npd.FOOD_PER_ROUND, (12 - 2) * 12)
    assert torch.equal(state.food_flat_positions, first_positions)
    assert torch.equal(state.food_grid, first_food)
    empty_food = state.food_grid.bool() & state.index_grid.eq(-1)
    assert state.grid[empty_food].eq(npd.FOOD_INPUT_VALUE).all()


def test_tensor_rank1_move_into_food_consumes_it_and_rewards_health_cpu():
    torch.manual_seed(123)
    state = TensorRank1State.random(
        cells=1,
        height=12,
        width=12,
        families=1,
        device=torch.device('cpu'),
        initial_health=10,
    )
    food_flat = state.food_flat_positions[
        state.food_flat_positions.remainder(state.grid_stride) > 2
    ][0]
    start_flat = food_flat - 1
    state.flat_positions[0] = start_flat
    state.positions[0, 0] = start_flat.div(state.grid_stride, rounding_mode='floor')
    state.positions[0, 1] = start_flat.remainder(state.grid_stride)
    state.health[0] = 10
    state.rebuild_grids()
    state.spawn_fixed_food()

    state.base_weight_1.zero_()
    state.base_weight_2.zero_()
    state.refresh_base_weight_matmul_cache()
    state.u_1.zero_()
    state.v_1.zero_()
    state.u_2.zero_()
    state.v_2.zero_()
    state.coeff_1.zero_()
    state.coeff_2.zero_()
    state.bias_1.zero_()
    state.bias_2.fill_(-100)
    state.bias_2[:, 3] = 100
    state.bias_2[:, npd.ATTACK_OUTPUT_INDEX] = -100

    actions = state.step(movement='snapshot_combat', compact_dead=False, sync_positions=False)

    assert int(actions[0].item()) == 3
    assert int(state.flat_positions[0].item()) == int(food_flat.item())
    assert state.health[0].item() == pytest.approx(10 - npd.MOVEMENT_HEALTH_COST + npd.FOOD_HEALTH_REWARD)
    assert int(state.food_grid.reshape(-1)[food_flat].item()) == 0
    assert int(state.grid.reshape(-1)[food_flat].item()) == 1


def test_tensor_rank1_benchmark_refills_empty_board_on_cpu():
    torch.manual_seed(123)
    metrics = benchmark_tensor_state(
        cells=0,
        height=8,
        width=8,
        families=1,
        steps=3,
        warmup_steps=0,
        movement='snapshot_combat',
        device=torch.device('cpu'),
        initial_health=1,
        wave_size=4,
        wave_initial_health=1,
        compact_every=2,
        npc_count=0,
    )

    assert metrics['compact_every'] == 2
    assert metrics['empty_refills'] > 0
    assert metrics['waves_spawned'] > 0
    assert metrics['processed_cell_steps'] > 0


def test_tensor_rank1_benchmark_can_disable_checksum_cpu():
    torch.manual_seed(123)
    metrics = benchmark_tensor_state(
        cells=8,
        height=8,
        width=8,
        families=1,
        steps=3,
        warmup_steps=0,
        movement='none',
        device=torch.device('cpu'),
        checksum_actions=0,
    )

    assert metrics['checksum_actions'] == 0
    assert metrics['action_checksum'] is None
    assert metrics['processed_cell_steps'] > 0


def test_tensor_rank1_benchmark_can_trace_segments_cpu():
    torch.manual_seed(123)
    metrics = benchmark_tensor_state(
        cells=8,
        height=8,
        width=8,
        families=1,
        steps=4,
        warmup_steps=0,
        movement='none',
        device=torch.device('cpu'),
        checksum_actions=0,
        trace_every=2,
    )

    assert metrics['trace_every'] == 2
    assert len(metrics['trace_segments']) == 2
    assert metrics['trace_segments'][0]['start_step'] == 0
    assert metrics['trace_segments'][0]['end_step'] == 2
    assert metrics['trace_segments'][0]['processed_cell_steps'] > 0
    assert metrics['trace_segments'][0]['active_cells_start'] == 8
    assert metrics['trace_segments'][0]['active_cells_end'] == 8


def test_tensor_rank1_benchmark_static_capacity_cpu():
    torch.manual_seed(123)
    metrics = benchmark_tensor_state(
        cells=8,
        height=8,
        width=8,
        families=1,
        steps=3,
        warmup_steps=0,
        movement='snapshot_combat',
        device=torch.device('cpu'),
        initial_health=20,
        wave_every=1,
        wave_size=2,
        wave_initial_health=20,
        compact_every=0,
        checksum_actions=0,
        static_capacity=True,
        cell_capacity=20,
        family_capacity=5,
    )

    assert metrics['static_capacity'] is True
    assert metrics['cell_capacity'] == 20
    assert metrics['cells_final'] == 20
    assert metrics['active_cells_final'] >= 8
    assert metrics['families_final'] == 5
    assert metrics['active_families_final'] == 4
    assert metrics['waves_spawned'] > 0


def test_tensor_rank1_benchmark_reports_health_dtype_cpu():
    torch.manual_seed(123)
    metrics = benchmark_tensor_state(
        cells=8,
        height=8,
        width=8,
        families=1,
        steps=2,
        warmup_steps=0,
        movement='snapshot_combat',
        device=torch.device('cpu'),
        initial_health=5,
        compact_every=0,
        checksum_actions=0,
        static_capacity=True,
        family_capacity=3,
        health_dtype='int32',
    )

    assert metrics['health_dtype'] == 'int32'
    assert metrics['active_cells_final'] > 0


def test_tensor_rank1_static_rebuild_grid_requires_compile_cpu():
    with pytest.raises(ValueError, match='static_rebuild_grid requires compiled_step'):
        benchmark_tensor_state(
            cells=8,
            height=8,
            width=8,
            families=1,
            steps=1,
            warmup_steps=0,
            movement='snapshot_combat',
            device=torch.device('cpu'),
            compact_every=0,
            checksum_actions=0,
            static_capacity=True,
            family_capacity=3,
            static_rebuild_grid=True,
        )


def test_tensor_rank1_family_basis_step_requires_rebuild_grid_cpu():
    with pytest.raises(ValueError, match='family_basis_step requires static_rebuild_grid'):
        benchmark_tensor_state(
            cells=8,
            height=8,
            width=8,
            families=1,
            steps=1,
            warmup_steps=0,
            movement='snapshot_combat',
            device=torch.device('cpu'),
            compact_every=0,
            checksum_actions=0,
            static_capacity=True,
            family_capacity=3,
            compiled_step=True,
            family_basis_step=True,
        )


def test_tensor_rank1_invalid_compile_mode_rejected_cpu():
    with pytest.raises(ValueError, match='unsupported compile mode'):
        benchmark_tensor_state(
            cells=8,
            height=8,
            width=8,
            families=1,
            steps=1,
            warmup_steps=0,
            movement='none',
            device=torch.device('cpu'),
            checksum_actions=0,
            compile_mode='bad-mode',
        )


def test_tensor_rank1_compiled_block_steps_require_compile_cpu():
    with pytest.raises(ValueError, match='compiled_block_steps requires compiled_step'):
        benchmark_tensor_state(
            cells=8,
            height=8,
            width=8,
            families=1,
            steps=1,
            warmup_steps=0,
            movement='snapshot_combat',
            device=torch.device('cpu'),
            compact_every=0,
            checksum_actions=0,
            static_capacity=True,
            family_capacity=3,
            compiled_block_steps=2,
        )


def test_tensor_rank1_cuda_graph_block_requires_compiled_block_cpu():
    with pytest.raises(ValueError, match='cuda_graph_block requires compiled block steps'):
        benchmark_tensor_state(
            cells=8,
            height=8,
            width=8,
            families=1,
            steps=1,
            warmup_steps=0,
            movement='snapshot_combat',
            device=torch.device('cpu'),
            compact_every=0,
            checksum_actions=0,
            static_capacity=True,
            family_capacity=3,
            cuda_graph_block=True,
        )


def test_tensor_rank1_benchmark_static_capacity_empty_refill_cpu():
    torch.manual_seed(123)
    metrics = benchmark_tensor_state(
        cells=0,
        height=8,
        width=8,
        families=1,
        steps=2,
        warmup_steps=0,
        movement='snapshot_combat',
        device=torch.device('cpu'),
        initial_health=5,
        wave_size=4,
        wave_initial_health=5,
        compact_every=0,
        checksum_actions=0,
        static_capacity=True,
        family_capacity=3,
        static_refill_empty=True,
        npc_count=0,
    )

    assert metrics['static_refill_empty'] is True
    assert metrics['static_refill_check_every'] == 1
    assert metrics['empty_refills'] == 1
    assert metrics['waves_spawned'] == 4
    assert metrics['active_cells_final'] > 0


def test_tensor_rank1_static_empty_refill_requires_family_capacity_cpu():
    with pytest.raises(ValueError, match='explicit family_capacity'):
        benchmark_tensor_state(
            cells=0,
            height=8,
            width=8,
            families=1,
            steps=1,
            warmup_steps=0,
            movement='snapshot_combat',
            device=torch.device('cpu'),
            wave_size=4,
            compact_every=0,
            checksum_actions=0,
            static_capacity=True,
            static_refill_empty=True,
        )


def test_tensor_rank1_static_capacity_requires_enough_cell_capacity_cpu():
    with pytest.raises(ValueError, match='cell_capacity must be at least cells'):
        benchmark_tensor_state(
            cells=8,
            height=8,
            width=8,
            families=1,
            steps=1,
            warmup_steps=0,
            movement='snapshot_combat',
            device=torch.device('cpu'),
            compact_every=0,
            checksum_actions=0,
            static_capacity=True,
            family_capacity=3,
            cell_capacity=4,
        )


def test_tensor_rank1_compiled_step_requires_cuda_cpu():
    with pytest.raises(ValueError, match='compiled_step requires a CUDA device'):
        benchmark_tensor_state(
            cells=8,
            height=8,
            width=8,
            families=1,
            steps=1,
            warmup_steps=0,
            movement='snapshot_combat',
            device=torch.device('cpu'),
            compact_every=0,
            checksum_actions=0,
            compiled_step=True,
        )


def test_tensor_rank1_empty_refill_uses_single_active_family_fast_path():
    torch.manual_seed(123)
    state = TensorRank1State.random(
        cells=0,
        height=8,
        width=8,
        families=1,
        device=torch.device('cpu'),
        npc_count=0,
    )

    spawned = state.append_weighted_wave(4, initial_health=5)

    assert spawned == 4
    assert state.single_active_family_id == 1
    assert state.family_index.eq(1).all()
    actions = state.step(movement='none')
    assert actions.shape == (4,)
    assert_tensor_state_position_invariants(state)


def test_tensor_rank1_benchmark_spawns_scheduled_waves_only_on_interval_cpu():
    torch.manual_seed(123)
    metrics = benchmark_tensor_state(
        cells=4,
        height=8,
        width=8,
        families=1,
        steps=5,
        warmup_steps=0,
        movement='none',
        device=torch.device('cpu'),
        initial_health=100,
        wave_every=2,
        wave_size=3,
        wave_initial_health=100,
        compact_every=2,
        npc_count=0,
    )

    assert metrics['empty_refills'] == 0
    assert metrics['waves_spawned'] == 6


def test_tensor_rank1_benchmark_normal_round_refill_uses_live_cell_count_cpu():
    torch.manual_seed(123)
    metrics = benchmark_tensor_state(
        cells=4,
        height=8,
        width=8,
        families=1,
        steps=4,
        warmup_steps=0,
        movement='none',
        device=torch.device('cpu'),
        initial_health=2,
        wave_every=2,
        wave_size=999,
        wave_initial_health=2,
        compact_every=2,
        normal_round_refill=True,
        per_wave=5,
        min_wave=2,
        npc_count=0,
    )

    assert metrics['normal_round_refill'] is True
    assert metrics['per_wave'] == 5
    assert metrics['min_wave'] == 2
    assert metrics['empty_refills'] == 0
    assert metrics['waves_spawned'] == 10
    assert metrics['active_cells_final'] == 5
