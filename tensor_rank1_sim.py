import time
from dataclasses import dataclass

import torch

import neural_petri_dish as npd


NEIGHBOR_INPUT_DIM = npd.NEIGHBOR_INPUT_DIM
INPUT_DIM = npd.NETWORK_INPUT_DIM
HIDDEN_DIM = npd.HIDDEN_DIM
DIRECTION_OUTPUT_DIM = npd.DIRECTION_OUTPUT_DIM
ATTACK_OUTPUT_INDEX = npd.ATTACK_OUTPUT_INDEX
OUTPUT_DIM = npd.OUTPUT_DIM
GRID_DTYPE = torch.int8
INDEX_GRID_DTYPE = torch.int32
MAX_HEALTH = 15
KILL_REWARD = npd.KILL_HEALTH_REWARD
FOOD_REWARD = npd.FOOD_HEALTH_REWARD
FOOD_INPUT_VALUE = npd.FOOD_INPUT_VALUE
BASE_ATTACK_DAMAGE = 1
LONE_TARGET_DAMAGE_BONUS = 1
_COMPILED_SNAPSHOT_COMBAT_STEP = {}
_COMPILED_REBUILD_SNAPSHOT_COMBAT_STEP = {}
_COMPILED_FAMILY_BASIS_REBUILD_SNAPSHOT_COMBAT_STEP = {}
_COMPILED_FAMILY_BASIS_REBUILD_SNAPSHOT_COMBAT_BLOCK = {}
HEALTH_DTYPES = {
    'float32': torch.float32,
    'int64': torch.long,
    'int32': torch.int32,
}
MATMUL_PRECISIONS = ('highest', 'high', 'medium')
COMPILE_MODES = ('default', 'reduce-overhead', 'max-autotune')
ACTIVATION_LIMIT = 1.0e6
LOGIT_LIMIT = 1.0e6


def refresh_runtime_constants():
    global KILL_REWARD, FOOD_REWARD, FOOD_INPUT_VALUE
    KILL_REWARD = npd.KILL_HEALTH_REWARD
    FOOD_REWARD = npd.FOOD_HEALTH_REWARD
    FOOD_INPUT_VALUE = npd.FOOD_INPUT_VALUE


def resolve_device(name):
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is false')
    return torch.device(name)


def resolve_health_dtype(name):
    try:
        return HEALTH_DTYPES[name]
    except KeyError as exc:
        raise ValueError(f'unsupported health dtype: {name}') from exc


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def reset_grid(grid):
    grid.zero_()
    grid[:, 0:2] = -1
    grid[:, -2:] = -1
    grid[0:2, :] = -1
    grid[-2:, :] = -1
    return grid


def make_grid(height, width, device):
    return reset_grid(torch.empty(height + 2, width + 4, device=device, dtype=GRID_DTYPE))


def reset_index_grid(index_grid):
    index_grid.fill_(-1)
    index_grid[:, 0:2] = -2
    index_grid[:, -2:] = -2
    index_grid[0:2, :] = -2
    index_grid[-2:, :] = -2
    return index_grid


def make_index_grid(height, width, device):
    return reset_index_grid(torch.empty(height + 2, width + 4, device=device, dtype=INDEX_GRID_DTYPE))


def make_food_grid(height, width, device):
    return torch.zeros(height + 2, width + 4, device=device, dtype=GRID_DTYPE)


def fixed_food_flat_positions(height, width, count, device):
    count = int(count)
    if count <= 0:
        return torch.empty(0, device=device, dtype=torch.long)
    playable_rows = max(int(height) - 2, 0)
    playable_cols = max(int(width), 0)
    if playable_rows == 0 or playable_cols == 0:
        return torch.empty(0, device=device, dtype=torch.long)

    rows = max(1, int(count ** 0.5))
    while rows > 1 and count % rows != 0:
        rows -= 1
    cols = max(1, (count + rows - 1) // rows)
    row_values = torch.linspace(2, int(height) - 1, rows, device=device).round().to(torch.long)
    col_values = torch.linspace(2, int(width) + 1, cols, device=device).round().to(torch.long)
    positions = torch.cartesian_prod(row_values, col_values)[:count]
    unique_positions = torch.unique(positions, dim=0)
    if unique_positions.shape[0] < min(count, playable_rows * playable_cols):
        all_positions = torch.cartesian_prod(
            torch.arange(2, int(height), device=device, dtype=torch.long),
            torch.arange(2, int(width) + 2, device=device, dtype=torch.long),
        )
        combined = torch.cat((unique_positions, all_positions), dim=0)
        unique_positions = torch.unique(combined, dim=0)[:count]
    return unique_positions[:, 0] * (int(width) + 4) + unique_positions[:, 1]


def refresh_food_overlay(grid, food_grid):
    playable_food = food_grid[2:-2, 2:-2]
    grid[2:-2, 2:-2].copy_(playable_food * FOOD_INPUT_VALUE)
    return grid


def clear_consumed_food(food_grid, target_flat_positions, consumed_food):
    if food_grid.numel() == 0:
        return
    food_flat = food_grid.reshape(-1)
    clear_flags = torch.zeros(food_flat.shape, device=food_grid.device, dtype=torch.int8)
    clear_flags.scatter_reduce_(
        0,
        target_flat_positions,
        consumed_food.to(clear_flags.dtype),
        reduce='amax',
        include_self=True,
    )
    food_flat.copy_(torch.where(clear_flags.bool(), torch.zeros_like(food_flat), food_flat))


def flatten_base_weight_1_for_matmul(base_weight_1):
    return base_weight_1.reshape(base_weight_1.shape[0] * HIDDEN_DIM, INPUT_DIM).t().contiguous()


def flatten_base_weight_2_for_matmul(base_weight_2):
    return base_weight_2.reshape(base_weight_2.shape[0] * OUTPUT_DIM, HIDDEN_DIM).t().contiguous()


def normalize_rank1_factors_(left, right):
    rms = torch.outer(left, right).square().mean().sqrt().clamp_min(torch.finfo(torch.float32).eps)
    scale = rms.sqrt()
    left.div_(scale)
    right.div_(scale)


def normalize_rank1_factor_rows_(left, right):
    rms = (
        left.square().mean(dim=1).sqrt()
        * right.square().mean(dim=1).sqrt()
    ).clamp_min(torch.finfo(torch.float32).eps)
    scale = rms.sqrt()
    left.div_(scale.unsqueeze(1))
    right.div_(scale.unsqueeze(1))
    return left, right


def init_linear_weight_bias(out_features, in_features, count, device):
    weights = torch.empty(count, out_features, in_features, device=device)
    biases = torch.empty(count, out_features, device=device)
    for index in range(count):
        torch.nn.init.kaiming_uniform_(weights[index], a=5 ** 0.5)
    bound = in_features ** -0.5
    torch.nn.init.uniform_(biases, -bound, bound)
    return weights, biases


def sanitize_recurrent_state(recurrent_state):
    return torch.nan_to_num(recurrent_state, nan=0.0, posinf=ACTIVATION_LIMIT, neginf=0.0)


def stabilize_hidden(hidden):
    return hidden.clamp_(0.0, ACTIVATION_LIMIT)


def stabilize_logits(logits):
    return logits.clamp_(-LOGIT_LIMIT, LOGIT_LIMIT)


def apply_movement_health_cost(health, moved):
    if npd.MOVEMENT_HEALTH_COST <= 0:
        return health
    cost = torch.as_tensor(npd.MOVEMENT_HEALTH_COST, device=health.device, dtype=torch.float32)
    return health - moved.to(torch.float32) * cost


def apply_stationary_health_cost(health, stationary_steps, stationary_health_cost):
    damage_stationary = (stationary_health_cost > 0) & (stationary_steps >= npd.STATIONARY_DAMAGE_AFTER_STEPS)
    cost = stationary_health_cost.to(health.dtype)
    return torch.where(damage_stationary, health - cost, health)


def packed_actions_from_logits(logits):
    directions = logits[:, :DIRECTION_OUTPUT_DIM].argmax(dim=1)
    attacks = logits[:, ATTACK_OUTPUT_INDEX] > 0
    return directions + attacks.to(directions.dtype) * DIRECTION_OUTPUT_DIM


def action_directions(actions):
    return actions.remainder(DIRECTION_OUTPUT_DIM)


def action_attack_intents(actions):
    return actions >= DIRECTION_OUTPUT_DIM


def snapshot_attack_damage(hits_occupied, target_flat_positions, attacker_indices, index_flat, direction_flat_deltas, dtype):
    has_other_neighbor = torch.zeros_like(hits_occupied)
    for offset_index in range(1, 9):
        neighbor_values = index_flat[target_flat_positions + direction_flat_deltas[offset_index]]
        has_other_neighbor = has_other_neighbor | ((neighbor_values >= 0) & (neighbor_values != attacker_indices))
    lone_target = hits_occupied & ~has_other_neighbor
    return (
        torch.full(hits_occupied.shape, BASE_ATTACK_DAMAGE, device=hits_occupied.device, dtype=dtype)
        + lone_target.to(dtype) * LONE_TARGET_DAMAGE_BONUS
    )


def snapshot_combat_step_tensors(
        grid,
        index_grid,
        food_grid,
        flat_positions,
        health,
        recurrent_state,
        family_index,
        coeff_1,
        coeff_2,
        bias_1,
        bias_2,
        base_weight_1,
        base_weight_2,
        u_1,
        v_1,
        u_2,
        v_2,
        neighbor_flat_offsets,
        direction_flat_deltas):
    inputs = torch.empty(flat_positions.shape[0], INPUT_DIM, device=grid.device)
    neighbor_indices = flat_positions[:, None] + neighbor_flat_offsets[None, :]
    inputs[:, :NEIGHBOR_INPUT_DIM] = grid.reshape(-1)[neighbor_indices]
    inputs[:, NEIGHBOR_INPUT_DIM:] = sanitize_recurrent_state(recurrent_state)

    selected_base_weight_1 = base_weight_1[family_index]
    base_hidden = torch.bmm(selected_base_weight_1, inputs.unsqueeze(2)).squeeze(2)
    rank1_hidden_scale = (inputs * v_1).sum(dim=1) * coeff_1
    hidden = stabilize_hidden(torch.relu(base_hidden + rank1_hidden_scale.unsqueeze(1) * u_1 + bias_1))

    selected_base_weight_2 = base_weight_2[family_index]
    base_logits = torch.bmm(selected_base_weight_2, hidden.unsqueeze(2)).squeeze(2)
    rank1_logit_scale = (hidden * v_2).sum(dim=1) * coeff_2
    logits = stabilize_logits(base_logits + rank1_logit_scale.unsqueeze(1) * u_2 + bias_2)
    actions = packed_actions_from_logits(logits)

    old_flat_positions = flat_positions
    directions = action_directions(actions)
    attack_intents = action_attack_intents(actions)
    target_flat_positions = flat_positions + direction_flat_deltas[directions]
    index_flat = index_grid.reshape(-1)
    food_flat = food_grid.reshape(-1)
    target_indices = index_flat[target_flat_positions]
    active = health > 0
    targeted = active & (directions != 0)
    move_intents = targeted & ~attack_intents
    attack_intents = targeted & attack_intents
    hits_border = move_intents & (target_indices == -2)
    hits_empty = move_intents & (target_indices == -1)
    hits_occupied = attack_intents & (target_indices >= 0)

    valid_targets = target_indices.clamp_min(0).to(torch.long)
    attacker_indices = torch.arange(flat_positions.shape[0], device=grid.device, dtype=index_grid.dtype)
    attack_damage = snapshot_attack_damage(
        hits_occupied,
        target_flat_positions,
        attacker_indices,
        index_grid.reshape(-1),
        direction_flat_deltas,
        health.dtype,
    )
    damage_received = torch.zeros_like(health)
    damage_received.scatter_add_(0, valid_targets, attack_damage * hits_occupied.to(health.dtype))

    target_health_after = health[valid_targets] - damage_received[valid_targets]
    target_survives = hits_occupied & (target_health_after > 0)
    target_killed = hits_occupied & (target_health_after <= 0)
    attacker_penalty = (hits_border | target_survives).to(health.dtype)
    attacker_reward = target_killed.to(health.dtype) * KILL_REWARD
    new_health = (health - damage_received - attacker_penalty + attacker_reward).clamp_max(MAX_HEALTH)
    new_health = apply_movement_health_cost(new_health, hits_empty)
    new_flat_positions = torch.where(hits_empty | target_killed, target_flat_positions, flat_positions)

    alive = new_health > 0
    grid_flat = grid.reshape(-1)
    grid_flat[old_flat_positions] = food_flat[old_flat_positions] * FOOD_INPUT_VALUE
    index_flat[old_flat_positions] = -1
    indices = attacker_indices
    write_indices = torch.where(alive, indices, torch.full_like(indices, -1))
    index_flat.scatter_reduce_(
        0,
        new_flat_positions,
        write_indices,
        reduce='amax',
        include_self=True,
    )
    owns_position = index_flat[new_flat_positions] == indices
    consumed_food = alive & owns_position & hits_empty & (food_flat[target_flat_positions] > 0)
    new_health = torch.where(
        consumed_food,
        (new_health + torch.as_tensor(FOOD_REWARD, device=grid.device, dtype=health.dtype)).clamp_max(MAX_HEALTH),
        new_health,
    )
    clear_consumed_food(food_grid, target_flat_positions, consumed_food)
    new_health = torch.where(alive & owns_position, new_health, torch.zeros_like(new_health))
    refresh_food_overlay(grid, food_grid)
    grid_flat[new_flat_positions] = (index_flat[new_flat_positions] >= 0).to(grid.dtype)
    return new_flat_positions, new_health, hidden, actions


def snapshot_combat_step_tensors_rebuild_grid(
        grid,
        index_grid,
        food_grid,
        flat_positions,
        health,
        recurrent_state,
        family_index,
        coeff_1,
        coeff_2,
        bias_1,
        bias_2,
        base_weight_1,
        base_weight_2,
        u_1,
        v_1,
        u_2,
        v_2,
        neighbor_flat_offsets,
        direction_flat_deltas):
    inputs = torch.empty(flat_positions.shape[0], INPUT_DIM, device=grid.device)
    neighbor_indices = flat_positions[:, None] + neighbor_flat_offsets[None, :]
    inputs[:, :NEIGHBOR_INPUT_DIM] = grid.reshape(-1)[neighbor_indices]
    inputs[:, NEIGHBOR_INPUT_DIM:] = sanitize_recurrent_state(recurrent_state)

    selected_base_weight_1 = base_weight_1[family_index]
    base_hidden = torch.bmm(selected_base_weight_1, inputs.unsqueeze(2)).squeeze(2)
    rank1_hidden_scale = (inputs * v_1).sum(dim=1) * coeff_1
    hidden = stabilize_hidden(torch.relu(base_hidden + rank1_hidden_scale.unsqueeze(1) * u_1 + bias_1))

    selected_base_weight_2 = base_weight_2[family_index]
    base_logits = torch.bmm(selected_base_weight_2, hidden.unsqueeze(2)).squeeze(2)
    rank1_logit_scale = (hidden * v_2).sum(dim=1) * coeff_2
    logits = stabilize_logits(base_logits + rank1_logit_scale.unsqueeze(1) * u_2 + bias_2)
    actions = packed_actions_from_logits(logits)

    directions = action_directions(actions)
    attack_intents = action_attack_intents(actions)
    target_flat_positions = flat_positions + direction_flat_deltas[directions]
    index_flat = index_grid.reshape(-1)
    food_flat = food_grid.reshape(-1)
    target_indices = index_flat[target_flat_positions]
    active = health > 0
    targeted = active & (directions != 0)
    move_intents = targeted & ~attack_intents
    attack_intents = targeted & attack_intents
    hits_border = move_intents & (target_indices == -2)
    hits_empty = move_intents & (target_indices == -1)
    hits_occupied = attack_intents & (target_indices >= 0)

    valid_targets = target_indices.clamp_min(0).to(torch.long)
    attacker_indices = torch.arange(flat_positions.shape[0], device=grid.device, dtype=index_grid.dtype)
    attack_damage = snapshot_attack_damage(
        hits_occupied,
        target_flat_positions,
        attacker_indices,
        index_grid.reshape(-1),
        direction_flat_deltas,
        health.dtype,
    )
    damage_received = torch.zeros_like(health)
    damage_received.scatter_add_(0, valid_targets, attack_damage * hits_occupied.to(health.dtype))

    target_health_after = health[valid_targets] - damage_received[valid_targets]
    target_survives = hits_occupied & (target_health_after > 0)
    target_killed = hits_occupied & (target_health_after <= 0)
    attacker_penalty = (hits_border | target_survives).to(health.dtype)
    attacker_reward = target_killed.to(health.dtype) * KILL_REWARD
    new_health = (health - damage_received - attacker_penalty + attacker_reward).clamp_max(MAX_HEALTH)
    new_health = apply_movement_health_cost(new_health, hits_empty)
    new_flat_positions = torch.where(hits_empty | target_killed, target_flat_positions, flat_positions)

    alive = new_health > 0
    index_grid[2:-2, 2:-2].fill_(-1)
    refresh_food_overlay(grid, food_grid)
    grid_flat = grid.reshape(-1)
    indices = attacker_indices
    write_indices = torch.where(alive, indices, torch.full_like(indices, -1))
    index_flat.scatter_reduce_(
        0,
        new_flat_positions,
        write_indices,
        reduce='amax',
        include_self=True,
    )
    owns_position = index_flat[new_flat_positions] == indices
    consumed_food = alive & owns_position & hits_empty & (food_flat[target_flat_positions] > 0)
    new_health = torch.where(
        consumed_food,
        (new_health + torch.as_tensor(FOOD_REWARD, device=grid.device, dtype=health.dtype)).clamp_max(MAX_HEALTH),
        new_health,
    )
    clear_consumed_food(food_grid, target_flat_positions, consumed_food)
    new_health = torch.where(alive & owns_position, new_health, torch.zeros_like(new_health))
    refresh_food_overlay(grid, food_grid)
    grid_flat[new_flat_positions] = (index_flat[new_flat_positions] >= 0).to(grid.dtype)
    return new_flat_positions, new_health, hidden, actions


def snapshot_combat_step_tensors_family_basis_rebuild_grid(
        index_grid,
        food_grid,
        flat_positions,
        health,
        stationary_steps,
        recurrent_state,
        round_survival_steps,
        family_index,
        coeff_1,
        coeff_2,
        bias_1,
        bias_2,
        base_weight_1_matmul,
        base_weight_2_matmul,
        u_1,
        v_1,
        u_2,
        v_2,
        stationary_health_cap,
        scatter_indices,
        dead_scatter_indices,
        neighbor_flat_offsets,
        direction_flat_deltas):
    active = health > 0
    round_survival_steps = round_survival_steps + active.to(round_survival_steps.dtype)
    inputs = torch.empty(flat_positions.shape[0], INPUT_DIM, device=index_grid.device)
    neighbor_indices = flat_positions[:, None] + neighbor_flat_offsets[None, :]
    index_flat = index_grid.reshape(-1)
    # The family-basis compiled path treats index_grid as authoritative and
    # derives the same ternary neighbor encoding as grid: border=-1, empty=0,
    # occupied=1. This avoids maintaining a duplicate binary occupancy grid.
    neighbor_values = index_flat[neighbor_indices]
    food_flat = food_grid.reshape(-1)
    neighbor_food = food_flat[neighbor_indices] > 0
    empty_food = (neighbor_values == -1) & neighbor_food
    inputs[:, :NEIGHBOR_INPUT_DIM] = (
        (neighbor_values >= 0).to(inputs.dtype)
        - (neighbor_values == -2).to(inputs.dtype)
        + empty_food.to(inputs.dtype) * FOOD_INPUT_VALUE
    )
    inputs[:, NEIGHBOR_INPUT_DIM:] = sanitize_recurrent_state(recurrent_state)

    family_count = base_weight_1_matmul.shape[1] // HIDDEN_DIM
    base_hidden_flat = inputs.matmul(base_weight_1_matmul)
    base_hidden_all = base_hidden_flat.reshape(flat_positions.shape[0], family_count, HIDDEN_DIM)
    hidden_family_selector = family_index.reshape(-1, 1, 1).expand(-1, 1, HIDDEN_DIM)
    base_hidden = base_hidden_all.gather(1, hidden_family_selector).squeeze(1)
    rank1_hidden_scale = (inputs * v_1).sum(dim=1) * coeff_1
    hidden = stabilize_hidden(torch.relu(base_hidden + rank1_hidden_scale.unsqueeze(1) * u_1 + bias_1))

    base_logits_flat = hidden.matmul(base_weight_2_matmul)
    base_logits_all = base_logits_flat.reshape(flat_positions.shape[0], family_count, OUTPUT_DIM)
    output_family_selector = family_index.reshape(-1, 1, 1).expand(-1, 1, OUTPUT_DIM)
    base_logits = base_logits_all.gather(1, output_family_selector).squeeze(1)
    rank1_logit_scale = (hidden * v_2).sum(dim=1) * coeff_2
    logits = stabilize_logits(base_logits + rank1_logit_scale.unsqueeze(1) * u_2 + bias_2)
    actions = packed_actions_from_logits(logits)

    directions = action_directions(actions)
    attack_intents = action_attack_intents(actions)
    target_flat_positions = flat_positions + direction_flat_deltas[directions]
    target_indices = index_flat[target_flat_positions]
    targeted = active & (directions != 0)
    move_intents = targeted & ~attack_intents
    attack_intents = targeted & attack_intents
    hits_border = move_intents & (target_indices == -2)
    hits_empty = move_intents & (target_indices == -1)
    hits_occupied = attack_intents & (target_indices >= 0)

    valid_targets = target_indices.clamp_min(0).to(torch.long)
    attack_damage = snapshot_attack_damage(
        hits_occupied,
        target_flat_positions,
        scatter_indices,
        index_flat,
        direction_flat_deltas,
        health.dtype,
    )
    damage_received = torch.zeros_like(health)
    damage_received.scatter_add_(0, valid_targets, attack_damage * hits_occupied.to(health.dtype))

    target_health_after = health[valid_targets] - damage_received[valid_targets]
    target_survives = hits_occupied & (target_health_after > 0)
    target_killed = hits_occupied & (target_health_after <= 0)
    lone_target_hits = hits_occupied & (attack_damage > BASE_ATTACK_DAMAGE)
    attacker_penalty = (hits_border | target_survives).to(health.dtype)
    attacker_reward = target_killed.to(health.dtype) * KILL_REWARD
    new_health = (health - damage_received - attacker_penalty + attacker_reward).clamp_max(MAX_HEALTH)
    new_health = apply_movement_health_cost(new_health, hits_empty)
    new_flat_positions = torch.where(hits_empty | target_killed, target_flat_positions, flat_positions)
    stayed_put = active & (new_flat_positions == flat_positions)
    new_stationary_steps = torch.where(stayed_put, stationary_steps + 1, torch.zeros_like(stationary_steps))
    new_health = apply_stationary_health_cost(new_health, new_stationary_steps, stationary_health_cap)
    new_stationary_steps = torch.where(new_health > 0, new_stationary_steps, torch.zeros_like(new_stationary_steps))

    alive = new_health > 0
    index_grid[2:-2, 2:-2].fill_(-1)
    write_indices = torch.where(alive, scatter_indices, dead_scatter_indices)
    index_flat.scatter_reduce_(
        0,
        new_flat_positions,
        write_indices,
        reduce='amax',
        include_self=True,
    )
    owns_position = index_flat[new_flat_positions] == scatter_indices
    consumed_food = alive & owns_position & hits_empty & (food_flat[target_flat_positions] > 0)
    move_success = alive & owns_position & hits_empty
    new_health = torch.where(
        consumed_food,
        (new_health + torch.as_tensor(FOOD_REWARD, device=index_grid.device, dtype=health.dtype)).clamp_max(MAX_HEALTH),
        new_health,
    )
    clear_consumed_food(food_grid, target_flat_positions, consumed_food)
    new_health = torch.where(alive & owns_position, new_health, torch.zeros_like(new_health))
    new_stationary_steps = torch.where(
        new_health > 0,
        new_stationary_steps,
        torch.zeros_like(new_stationary_steps),
    )
    event_counts = torch.stack((
        active.sum(),
        move_intents.sum(),
        move_success.sum(),
        attack_intents.sum(),
        hits_occupied.sum(),
        target_killed.sum(),
        lone_target_hits.sum(),
        hits_border.sum(),
        consumed_food.sum(),
        (active & (new_health <= 0)).sum(),
        stayed_put.sum(),
        (new_health > 0).sum(),
    )).to(torch.float32)
    return new_flat_positions, new_health, new_stationary_steps, hidden, actions, round_survival_steps, event_counts


def resolve_compile_mode(mode):
    if mode not in COMPILE_MODES:
        raise ValueError(f'unsupported compile mode: {mode}')
    return mode


def compiled_snapshot_combat_step_tensors(mode='reduce-overhead'):
    mode = resolve_compile_mode(mode)
    if mode not in _COMPILED_SNAPSHOT_COMBAT_STEP:
        _COMPILED_SNAPSHOT_COMBAT_STEP[mode] = torch.compile(
            snapshot_combat_step_tensors,
            mode=mode,
        )
    return _COMPILED_SNAPSHOT_COMBAT_STEP[mode]


def compiled_rebuild_snapshot_combat_step_tensors(mode='reduce-overhead'):
    mode = resolve_compile_mode(mode)
    if mode not in _COMPILED_REBUILD_SNAPSHOT_COMBAT_STEP:
        _COMPILED_REBUILD_SNAPSHOT_COMBAT_STEP[mode] = torch.compile(
            snapshot_combat_step_tensors_rebuild_grid,
            mode=mode,
        )
    return _COMPILED_REBUILD_SNAPSHOT_COMBAT_STEP[mode]


def compiled_family_basis_rebuild_snapshot_combat_step_tensors(mode='reduce-overhead'):
    mode = resolve_compile_mode(mode)
    if mode not in _COMPILED_FAMILY_BASIS_REBUILD_SNAPSHOT_COMBAT_STEP:
        _COMPILED_FAMILY_BASIS_REBUILD_SNAPSHOT_COMBAT_STEP[mode] = torch.compile(
            snapshot_combat_step_tensors_family_basis_rebuild_grid,
            mode=mode,
        )
    return _COMPILED_FAMILY_BASIS_REBUILD_SNAPSHOT_COMBAT_STEP[mode]


def family_basis_rebuild_snapshot_combat_block_tensors(block_steps):
    block_steps = int(block_steps)

    def block_fn(
            index_grid,
            food_grid,
            flat_positions,
            health,
            stationary_steps,
            recurrent_state,
            round_survival_steps,
            family_index,
            coeff_1,
            coeff_2,
            bias_1,
            bias_2,
            base_weight_1_matmul,
            base_weight_2_matmul,
            u_1,
            v_1,
            u_2,
            v_2,
            stationary_health_cap,
            scatter_indices,
            dead_scatter_indices,
            neighbor_flat_offsets,
            direction_flat_deltas):
        total_event_counts = torch.zeros(12, device=health.device, dtype=torch.float32)
        for _ in range(block_steps):
            flat_positions, health, stationary_steps, recurrent_state, _actions, round_survival_steps, event_counts = snapshot_combat_step_tensors_family_basis_rebuild_grid(
                index_grid,
                food_grid,
                flat_positions,
                health,
                stationary_steps,
                recurrent_state,
                round_survival_steps,
                family_index,
                coeff_1,
                coeff_2,
                bias_1,
                bias_2,
                base_weight_1_matmul,
                base_weight_2_matmul,
                u_1,
                v_1,
                u_2,
                v_2,
                stationary_health_cap,
                scatter_indices,
                dead_scatter_indices,
                neighbor_flat_offsets,
                direction_flat_deltas,
            )
            total_event_counts = total_event_counts + event_counts
        return flat_positions, health, stationary_steps, recurrent_state, round_survival_steps, total_event_counts

    return block_fn


def compiled_family_basis_rebuild_snapshot_combat_block_tensors(block_steps, mode='default'):
    mode = resolve_compile_mode(mode)
    block_steps = int(block_steps)
    if block_steps <= 0:
        raise ValueError('block_steps must be positive')
    key = (mode, block_steps)
    if key not in _COMPILED_FAMILY_BASIS_REBUILD_SNAPSHOT_COMBAT_BLOCK:
        _COMPILED_FAMILY_BASIS_REBUILD_SNAPSHOT_COMBAT_BLOCK[key] = torch.compile(
            family_basis_rebuild_snapshot_combat_block_tensors(block_steps),
            mode=mode,
        )
    return _COMPILED_FAMILY_BASIS_REBUILD_SNAPSHOT_COMBAT_BLOCK[key]


class CudaGraphFamilyBasisBlockRunner:
    def __init__(self, state, block_steps, compile_mode='default'):
        if state.device.type != 'cuda':
            raise ValueError('cuda graph block replay requires CUDA tensors')
        self.state = state
        self.block_steps = int(block_steps)
        self.step_fn = compiled_family_basis_rebuild_snapshot_combat_block_tensors(
            self.block_steps,
            compile_mode,
        )
        self.graph = torch.cuda.CUDAGraph()
        self.event_counts = torch.empty(12, device=state.device, dtype=torch.float32)
        original_index_grid = state.index_grid.clone()
        original_food_grid = state.food_grid.clone()
        original_flat_positions = state.flat_positions.clone()
        original_health = state.health.clone()
        original_stationary_steps = state.stationary_steps.clone()
        original_recurrent_state = state.recurrent_state.clone()
        original_round_survival_steps = state.round_survival_steps.clone()

        with torch.cuda.graph(self.graph):
            flat_positions, health, stationary_steps, recurrent_state, round_survival_steps, event_counts = self.step_fn(
                *state.family_basis_block_args()
            )
            state.flat_positions.copy_(flat_positions)
            state.health.copy_(health)
            state.stationary_steps.copy_(stationary_steps)
            state.recurrent_state.copy_(recurrent_state)
            state.round_survival_steps.copy_(round_survival_steps)
            self.event_counts.copy_(event_counts)
        state.index_grid.copy_(original_index_grid)
        state.food_grid.copy_(original_food_grid)
        state.flat_positions.copy_(original_flat_positions)
        state.health.copy_(original_health)
        state.stationary_steps.copy_(original_stationary_steps)
        state.recurrent_state.copy_(original_recurrent_state)
        state.round_survival_steps.copy_(original_round_survival_steps)
        synchronize(state.device)

    def replay(self):
        self.graph.replay()
        return self.event_counts




@dataclass
class TensorRank1State:
    grid: torch.Tensor
    index_grid: torch.Tensor
    food_grid: torch.Tensor
    food_flat_positions: torch.Tensor
    positions: torch.Tensor
    flat_positions: torch.Tensor
    health: torch.Tensor
    stationary_steps: torch.Tensor
    recurrent_state: torch.Tensor
    family_index: torch.Tensor
    coeff_1: torch.Tensor
    coeff_2: torch.Tensor
    bias_1: torch.Tensor
    bias_2: torch.Tensor
    base_weight_1: torch.Tensor
    base_weight_2: torch.Tensor
    base_weight_1_matmul: torch.Tensor
    base_weight_2_matmul: torch.Tensor
    u_1: torch.Tensor
    v_1: torch.Tensor
    u_2: torch.Tensor
    v_2: torch.Tensor
    stationary_health_cap: torch.Tensor
    neighbor_offsets: torch.Tensor
    neighbor_flat_offsets: torch.Tensor
    direction_deltas: torch.Tensor
    direction_flat_deltas: torch.Tensor
    round_survival_steps: torch.Tensor = None
    round_participants: torch.Tensor = None
    evolution_anchor_weight_1: torch.Tensor = None
    evolution_anchor_bias_1: torch.Tensor = None
    evolution_anchor_weight_2: torch.Tensor = None
    evolution_anchor_bias_2: torch.Tensor = None
    single_active_family_id: int = None

    @classmethod
    def random(
            cls,
            cells,
            height,
            width,
            families,
            device,
            initial_health=2,
            health_dtype=torch.float32,
            coeff_scale=npd.FACTORED_WAVE_COEFF_SCALE,
            stationary_health_cap=1):
        device = torch.device(device)
        playable_rows = height - 2
        playable_cols = width
        if playable_rows <= 0:
            raise ValueError('height must leave room for the two-cell border')
        if cells > playable_rows * playable_cols:
            raise ValueError('cells must not exceed playable grid positions')
        grid = make_grid(height, width, device)
        index_grid = make_index_grid(height, width, device)
        food_grid = make_food_grid(height, width, device)
        food_flat_positions = fixed_food_flat_positions(height, width, npd.FOOD_PER_ROUND, device)
        flat_positions = torch.randperm(playable_rows * playable_cols, device=device)[:cells]
        positions = torch.empty(cells, 2, device=device, dtype=torch.long)
        positions[:, 0] = flat_positions.div(playable_cols, rounding_mode='floor') + 2
        positions[:, 1] = flat_positions.remainder(playable_cols) + 2
        grid_stride = width + 4
        grid_flat_positions = positions[:, 0] * grid_stride + positions[:, 1]
        indices = torch.arange(cells, device=device, dtype=index_grid.dtype)
        grid.reshape(-1)[grid_flat_positions] = 1
        index_grid.reshape(-1)[grid_flat_positions] = indices

        family_index = torch.randint(families, (cells,), device=device)
        health = torch.full((cells,), initial_health, device=device, dtype=health_dtype)
        stationary_steps = torch.zeros(cells, device=device, dtype=torch.int16)
        recurrent_state = torch.zeros(cells, HIDDEN_DIM, device=device)
        coeff_1 = torch.randn(cells, device=device) * coeff_scale
        coeff_2 = torch.randn(cells, device=device) * coeff_scale
        base_weight_1, family_bias_1 = init_linear_weight_bias(HIDDEN_DIM, INPUT_DIM, families, device)
        base_weight_2, family_bias_2 = init_linear_weight_bias(OUTPUT_DIM, HIDDEN_DIM, families, device)
        bias_1 = family_bias_1[family_index].clone()
        bias_2 = family_bias_2[family_index].clone()
        base_weight_1_matmul = flatten_base_weight_1_for_matmul(base_weight_1)
        base_weight_2_matmul = flatten_base_weight_2_for_matmul(base_weight_2)
        u_1 = torch.randn(cells, HIDDEN_DIM, device=device)
        v_1 = torch.randn(cells, INPUT_DIM, device=device)
        u_2 = torch.randn(cells, OUTPUT_DIM, device=device)
        v_2 = torch.randn(cells, HIDDEN_DIM, device=device)
        normalize_rank1_factor_rows_(u_1, v_1)
        normalize_rank1_factor_rows_(u_2, v_2)

        state = cls(
            grid=grid,
            index_grid=index_grid,
            food_grid=food_grid,
            food_flat_positions=food_flat_positions,
            positions=positions,
            flat_positions=grid_flat_positions,
            health=health,
            stationary_steps=stationary_steps,
            recurrent_state=recurrent_state,
            family_index=family_index,
            coeff_1=coeff_1,
            coeff_2=coeff_2,
            bias_1=bias_1,
            bias_2=bias_2,
            base_weight_1=base_weight_1,
            base_weight_2=base_weight_2,
            base_weight_1_matmul=base_weight_1_matmul,
            base_weight_2_matmul=base_weight_2_matmul,
            u_1=u_1,
            v_1=v_1,
            u_2=u_2,
            v_2=v_2,
            stationary_health_cap=torch.as_tensor(stationary_health_cap, device=device, dtype=health_dtype),
            neighbor_offsets=torch.as_tensor(npd.NEIGHBOR_OFFSETS, device=device, dtype=torch.long),
            neighbor_flat_offsets=torch.as_tensor(
                [dy * grid_stride + dx for dy, dx in npd.NEIGHBOR_OFFSETS],
                device=device,
                dtype=torch.long,
            ),
            direction_deltas=torch.as_tensor(npd.DIRECTION_DELTAS, device=device, dtype=torch.long),
            direction_flat_deltas=torch.as_tensor(
                [dy * grid_stride + dx for dy, dx in npd.DIRECTION_DELTAS],
                device=device,
                dtype=torch.long,
            ),
            round_survival_steps=torch.zeros(cells, device=device, dtype=torch.float32),
            round_participants=torch.ones(cells, device=device, dtype=torch.bool),
            evolution_anchor_weight_1=base_weight_1[0].clone(),
            evolution_anchor_bias_1=family_bias_1[0].clone(),
            evolution_anchor_weight_2=base_weight_2[0].clone(),
            evolution_anchor_bias_2=family_bias_2[0].clone(),
            single_active_family_id=0 if families == 1 else None,
        )
        state.spawn_fixed_food()
        return state

    @classmethod
    def fixed_capacity(
            cls,
            active_cells,
            height,
            width,
            active_families,
            family_capacity,
            device,
            initial_health=2,
            cell_capacity=None,
            health_dtype=torch.float32,
            coeff_scale=npd.FACTORED_WAVE_COEFF_SCALE,
            stationary_health_cap=1):
        board_capacity = (height - 2) * width
        capacity = board_capacity if cell_capacity is None else int(cell_capacity)
        if active_cells > capacity:
            raise ValueError('active_cells must not exceed cell_capacity')
        if capacity > board_capacity:
            raise ValueError('cell_capacity must not exceed playable grid capacity')
        if active_families > family_capacity:
            raise ValueError('active_families must not exceed family_capacity')
        state = cls.random(
            cells=capacity,
            height=height,
            width=width,
            families=active_families,
            device=device,
            initial_health=initial_health,
            health_dtype=health_dtype,
            coeff_scale=coeff_scale,
            stationary_health_cap=stationary_health_cap,
        )
        state.reserve_inactive_family_slots(family_capacity)
        if active_cells > 0:
            state.family_index[:active_cells] = torch.randint(
                active_families,
                (active_cells,),
                device=state.device,
                dtype=state.family_index.dtype,
            )
        if active_cells < capacity:
            inactive = torch.arange(active_cells, capacity, device=state.device)
            state.health[inactive] = 0
            state.round_participants[inactive] = False
            state.round_survival_steps[inactive] = 0
            state.family_index[inactive] = 0
            grid_flat = state.grid.reshape(-1)
            index_flat = state.index_grid.reshape(-1)
            grid_flat[state.flat_positions[inactive]] = 0
            index_flat[state.flat_positions[inactive]] = -1
            state.rebuild_grids()
        state.single_active_family_id = None
        return state

    @property
    def device(self):
        return self.grid.device

    def clone(self):
        kwargs = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            kwargs[field_name] = value.clone() if isinstance(value, torch.Tensor) else value
        return type(self)(**kwargs)

    def reserve_inactive_family_slots(self, family_capacity):
        family_capacity = int(family_capacity)
        if family_capacity <= self.families:
            return
        extra = family_capacity - self.families
        # Reserved rows are overwritten before they become live families, so
        # keeping them inert avoids making dynamics depend on unused capacity.
        self.base_weight_1 = torch.cat((
            self.base_weight_1,
            torch.zeros(
                extra,
                HIDDEN_DIM,
                INPUT_DIM,
                device=self.device,
                dtype=self.base_weight_1.dtype,
            ),
        ), dim=0)
        self.base_weight_2 = torch.cat((
            self.base_weight_2,
            torch.zeros(
                extra,
                OUTPUT_DIM,
                HIDDEN_DIM,
                device=self.device,
                dtype=self.base_weight_2.dtype,
            ),
        ), dim=0)
        self.refresh_base_weight_matmul_cache()

    def refresh_base_weight_matmul_cache(self):
        self.base_weight_1_matmul = flatten_base_weight_1_for_matmul(self.base_weight_1)
        self.base_weight_2_matmul = flatten_base_weight_2_for_matmul(self.base_weight_2)

    def refresh_base_weight_matmul_cache_row(self, family_id):
        family_id = int(family_id)
        hidden_start = family_id * HIDDEN_DIM
        output_start = family_id * OUTPUT_DIM
        self.base_weight_1_matmul[:, hidden_start:hidden_start + HIDDEN_DIM] = self.base_weight_1[family_id].t()
        self.base_weight_2_matmul[:, output_start:output_start + OUTPUT_DIM] = self.base_weight_2[family_id].t()

    @property
    def cells(self):
        return int(self.positions.shape[0])

    @property
    def families(self):
        return int(self.base_weight_1.shape[0])

    def family_capacity_version(self):
        return int(getattr(self, '_family_capacity_version', 0))

    def grow_family_capacity(self, new_capacity):
        new_capacity = int(new_capacity)
        old_capacity = self.families
        if new_capacity <= old_capacity:
            return
        extra = new_capacity - old_capacity
        self.base_weight_1 = torch.cat((
            self.base_weight_1,
            torch.randn(extra, HIDDEN_DIM, INPUT_DIM, device=self.device),
        ), dim=0)
        self.base_weight_2 = torch.cat((
            self.base_weight_2,
            torch.randn(extra, OUTPUT_DIM, HIDDEN_DIM, device=self.device),
        ), dim=0)
        self.refresh_base_weight_matmul_cache()
        self.single_active_family_id = None
        self._family_capacity_version = self.family_capacity_version() + 1

    @property
    def playable_shape(self):
        return self.grid.shape[0] - 4, self.grid.shape[1] - 4

    @property
    def grid_stride(self):
        return self.grid.shape[1]

    def sync_positions_from_flat(self):
        self.positions[:, 0] = self.flat_positions.div(self.grid_stride, rounding_mode='floor')
        self.positions[:, 1] = self.flat_positions.remainder(self.grid_stride)

    def cell_indices(self):
        cached = getattr(self, '_cell_indices', None)
        if cached is None or cached.shape[0] != self.cells or cached.device != self.device:
            cached = torch.arange(self.cells, device=self.device, dtype=torch.long)
            self._cell_indices = cached
        return cached

    def index_grid_indices(self):
        cached = getattr(self, '_index_grid_indices', None)
        if cached is None or cached.shape[0] != self.cells or cached.device != self.device:
            cached = torch.arange(self.cells, device=self.device, dtype=self.index_grid.dtype)
            self._index_grid_indices = cached
        return cached

    def dead_index_grid_indices(self):
        cached = getattr(self, '_dead_index_grid_indices', None)
        if cached is None or cached.shape[0] != self.cells or cached.device != self.device:
            cached = torch.full((self.cells,), -1, device=self.device, dtype=self.index_grid.dtype)
            self._dead_index_grid_indices = cached
        return cached

    def input_buffer(self):
        cached = getattr(self, '_input_buffer', None)
        if cached is None or cached.shape[0] != self.cells or cached.device != self.device:
            cached = torch.empty(self.cells, INPUT_DIM, device=self.device)
            self._input_buffer = cached
        return cached

    def damage_buffer(self):
        cached = getattr(self, '_damage_buffer', None)
        if (
                cached is None
                or cached.shape[0] != self.cells
                or cached.device != self.device
                or cached.dtype != self.health.dtype):
            cached = torch.empty(self.cells, device=self.device, dtype=self.health.dtype)
            self._damage_buffer = cached
        return cached

    def grid_index_write_buffer(self):
        cached = getattr(self, '_grid_index_write_buffer', None)
        if cached is None or cached.shape[0] != self.cells or cached.device != self.device:
            cached = torch.empty(self.cells, device=self.device, dtype=self.index_grid.dtype)
            self._grid_index_write_buffer = cached
        return cached

    def dead_mask_buffer(self):
        cached = getattr(self, '_dead_mask_buffer', None)
        if cached is None or cached.shape[0] != self.cells or cached.device != self.device:
            cached = torch.empty(self.cells, device=self.device, dtype=torch.bool)
            self._dead_mask_buffer = cached
        return cached

    def family_basis_block_args(self):
        return (
            self.index_grid,
            self.food_grid,
            self.flat_positions,
            self.health,
            self.stationary_steps,
            self.recurrent_state,
            self.round_survival_steps,
            self.family_index,
            self.coeff_1,
            self.coeff_2,
            self.bias_1,
            self.bias_2,
            self.base_weight_1_matmul,
            self.base_weight_2_matmul,
            self.u_1,
            self.v_1,
            self.u_2,
            self.v_2,
            self.stationary_health_cap,
            self.index_grid_indices(),
            self.dead_index_grid_indices(),
            self.neighbor_flat_offsets,
            self.direction_flat_deltas,
        )

    def alive_mask(self):
        return self.health > 0

    def spawn_fixed_food(self):
        self.food_grid.zero_()
        if self.food_flat_positions.numel() > 0:
            self.food_grid.reshape(-1)[self.food_flat_positions] = 1
        refresh_food_overlay(self.grid, self.food_grid)
        alive = self.alive_mask()
        if bool(alive.any()):
            self.grid.reshape(-1)[self.flat_positions[alive]] = 1
        return int(self.food_flat_positions.numel())

    def apply_round_transition_health_cost(self, compact_dead=False):
        alive = self.health > 0
        if not bool(alive.any()):
            return
        if npd.ROUND_TRANSITION_HEALTH_COST <= 0:
            return
        transition_cost = torch.as_tensor(
            npd.ROUND_TRANSITION_HEALTH_COST,
            device=self.device,
            dtype=self.health.dtype,
        )
        self.health = torch.where(alive, (self.health - transition_cost).clamp_min(0), self.health)
        self.stationary_steps = torch.where(
            self.health > 0,
            self.stationary_steps,
            torch.zeros_like(self.stationary_steps),
        )
        alive_after = self.health > 0
        if compact_dead:
            self.compact(alive_after)
        else:
            self.update_grids_incremental(self.flat_positions, alive=alive_after)

    def live_family_mask(self):
        used = torch.zeros(self.families, device=self.device, dtype=torch.bool)
        alive = self.alive_mask()
        if bool(alive.any()):
            used[self.family_index[alive]] = True
        return used

    def live_family_count(self):
        return int(self.live_family_mask().sum().item())

    def ensure_evolution_buffers(self):
        if self.round_survival_steps is None or self.round_survival_steps.shape[0] != self.cells:
            self.round_survival_steps = torch.zeros(self.cells, device=self.device, dtype=torch.float32)
        if self.round_participants is None or self.round_participants.shape[0] != self.cells:
            self.round_participants = self.health > 0
        if self.evolution_anchor_weight_1 is None:
            self.evolution_anchor_weight_1 = self.base_weight_1[0].clone()
            self.evolution_anchor_bias_1 = self.bias_1[0].clone()
            self.evolution_anchor_weight_2 = self.base_weight_2[0].clone()
            self.evolution_anchor_bias_2 = self.bias_2[0].clone()

    def record_survival_steps(self, step_count):
        self.ensure_evolution_buffers()
        alive = self.health > 0
        self.round_participants |= alive
        self.round_survival_steps[alive] += float(step_count)

    def reset_round_survival_tracking(self):
        self.ensure_evolution_buffers()
        self.round_survival_steps.zero_()
        self.round_participants.copy_(self.health > 0)

    def next_static_family_slot(self, family_count):
        family_count = int(family_count)
        used = self.live_family_mask()
        if family_count < self.families and not bool(used[family_count]):
            return family_count, family_count + 1

        reusable = torch.nonzero(~used, as_tuple=False).reshape(-1)
        if reusable.numel() == 0:
            family_id = self.families
            self.grow_family_capacity(max(self.families * 2, self.families + 1))
            return family_id, family_id + 1
        family_id = int(reusable[0].item())
        return family_id, max(family_count, family_id + 1)

    def gather_inputs(self):
        neighbor_indices = self.flat_positions[:, None] + self.neighbor_flat_offsets[None, :]
        neighbors = self.grid.reshape(-1)[neighbor_indices]
        inputs = self.input_buffer()
        inputs[:, :NEIGHBOR_INPUT_DIM] = neighbors
        inputs[:, NEIGHBOR_INPUT_DIM:] = sanitize_recurrent_state(self.recurrent_state)
        return inputs

    def forward_actions(self):
        inputs = self.gather_inputs()
        single_family_id = self.single_active_family_id
        if single_family_id is not None:
            base_hidden = inputs.matmul(self.base_weight_1[single_family_id].t())
            rank1_hidden_scale = (inputs * self.v_1).sum(dim=1) * self.coeff_1
            hidden = base_hidden.addcmul_(
                rank1_hidden_scale.unsqueeze(1),
                self.u_1,
            ).add_(self.bias_1).relu_()
            hidden = stabilize_hidden(hidden)
            base_logits = hidden.matmul(self.base_weight_2[single_family_id].t())
            rank1_logit_scale = (hidden * self.v_2).sum(dim=1) * self.coeff_2
            logits = base_logits.addcmul_(
                rank1_logit_scale.unsqueeze(1),
                self.u_2,
            )
        else:
            selected_base_weight_1 = self.base_weight_1[self.family_index]
            base_hidden = torch.bmm(selected_base_weight_1, inputs.unsqueeze(2)).squeeze(2)
            rank1_hidden_scale = (inputs * self.v_1).sum(dim=1) * self.coeff_1
            hidden = base_hidden.addcmul_(
                rank1_hidden_scale.unsqueeze(1),
                self.u_1,
            ).add_(self.bias_1).relu_()
            hidden = stabilize_hidden(hidden)
            selected_base_weight_2 = self.base_weight_2[self.family_index]
            base_logits = torch.bmm(selected_base_weight_2, hidden.unsqueeze(2)).squeeze(2)
            rank1_logit_scale = (hidden * self.v_2).sum(dim=1) * self.coeff_2
            logits = base_logits.addcmul_(
                rank1_logit_scale.unsqueeze(1),
                self.u_2,
            )
        logits.add_(self.bias_2)
        logits = stabilize_logits(logits)
        self.recurrent_state = hidden
        return packed_actions_from_logits(logits)

    def refresh_single_active_family(self):
        if self.cells == 0:
            self.single_active_family_id = None
            return
        first_family = self.family_index[0]
        if bool((self.family_index == first_family).all()):
            self.single_active_family_id = int(first_family.item())
        else:
            self.single_active_family_id = None

    def prune_unused_families(self):
        if self.cells == 0:
            self.single_active_family_id = None
            return
        used_families, inverse = torch.unique(self.family_index, sorted=True, return_inverse=True)
        if int(used_families.shape[0]) == self.families:
            self.refresh_single_active_family()
            return
        self.base_weight_1 = self.base_weight_1[used_families]
        self.base_weight_2 = self.base_weight_2[used_families]
        self.refresh_base_weight_matmul_cache()
        self.family_index = inverse.to(self.family_index.dtype)
        self.refresh_single_active_family()

    def apply_snapshot_movement(self, actions, sync_positions=True):
        old_flat_positions = self.flat_positions
        directions = action_directions(actions)
        target_flat_positions = self.flat_positions + self.direction_flat_deltas[directions]
        food_flat = self.food_grid.reshape(-1)
        target_indices = self.index_grid.reshape(-1)[target_flat_positions]
        can_move = (directions != 0) & (target_indices == -1) & ~action_attack_intents(actions)
        self.flat_positions = torch.where(can_move, target_flat_positions, self.flat_positions)
        health_after_move = apply_movement_health_cost(self.health, can_move)
        consumed_food = can_move & (food_flat[target_flat_positions] > 0)
        health_after_move = torch.where(
            consumed_food,
            (health_after_move + torch.as_tensor(FOOD_REWARD, device=self.device, dtype=self.health.dtype)).clamp_max(MAX_HEALTH),
            health_after_move,
        )
        clear_consumed_food(self.food_grid, target_flat_positions, consumed_food)
        if sync_positions:
            self.sync_positions_from_flat()
        self.update_grids_incremental(old_flat_positions)
        owns_position = self.index_grid.reshape(-1)[self.flat_positions] == self.index_grid_indices()
        self.health = torch.where(
            (health_after_move > 0) & owns_position,
            health_after_move,
            torch.zeros_like(health_after_move),
        )

    def update_grids_incremental(self, old_flat_positions, alive=None):
        grid_flat = self.grid.reshape(-1)
        index_flat = self.index_grid.reshape(-1)
        grid_flat[old_flat_positions] = self.food_grid.reshape(-1)[old_flat_positions] * FOOD_INPUT_VALUE
        index_flat[old_flat_positions] = -1
        if alive is None:
            indices = self.cell_indices().to(self.index_grid.dtype)
            live_flat_positions = self.flat_positions
            grid_flat[live_flat_positions] = 1
            index_flat[live_flat_positions] = indices
        else:
            indices = self.cell_indices().to(self.index_grid.dtype)
            write_indices = self.grid_index_write_buffer()
            dead = self.dead_mask_buffer()
            write_indices.copy_(indices)
            torch.logical_not(alive, out=dead)
            write_indices.masked_fill_(dead, -1)
            index_flat.scatter_reduce_(
                0,
                self.flat_positions,
                write_indices,
                reduce='amax',
                include_self=True,
            )
            grid_flat[self.flat_positions] = (index_flat[self.flat_positions] >= 0).to(self.grid.dtype)

    def rebuild_grids(self):
        reset_grid(self.grid)
        reset_index_grid(self.index_grid)
        refresh_food_overlay(self.grid, self.food_grid)
        grid_flat = self.grid.reshape(-1)
        index_flat = self.index_grid.reshape(-1)
        alive = self.alive_mask()
        indices = self.cell_indices()[alive].to(self.index_grid.dtype)
        live_flat_positions = self.flat_positions[alive]
        grid_flat[live_flat_positions] = 1
        index_flat[live_flat_positions] = indices

    def apply_snapshot_combat(self, actions, compact_dead=True, sync_positions=True):
        if self.cells == 0:
            return

        old_flat_positions = self.flat_positions
        directions = action_directions(actions)
        attack_intents = action_attack_intents(actions)
        target_flat_positions = self.flat_positions + self.direction_flat_deltas[directions]
        food_flat = self.food_grid.reshape(-1)
        target_indices = self.index_grid.reshape(-1)[target_flat_positions]
        active = self.alive_mask()
        targeted = active & (directions != 0)
        move_intents = targeted & ~attack_intents
        attack_intents = targeted & attack_intents
        hits_border = move_intents & (target_indices == -2)
        hits_empty = move_intents & (target_indices == -1)
        hits_occupied = attack_intents & (target_indices >= 0)
        valid_targets = target_indices.clamp_min(0).to(torch.long)
        attacker_indices = self.index_grid_indices()
        attack_damage = snapshot_attack_damage(
            hits_occupied,
            target_flat_positions,
            attacker_indices,
            self.index_grid.reshape(-1),
            self.direction_flat_deltas,
            self.health.dtype,
        )
        damage_received = self.damage_buffer()
        damage_received.zero_()
        damage_received.scatter_add_(0, valid_targets, attack_damage * hits_occupied.to(self.health.dtype))

        target_health_after = self.health[valid_targets] - damage_received[valid_targets]
        target_survives = hits_occupied & (target_health_after > 0)
        target_killed = hits_occupied & (target_health_after <= 0)
        attacker_penalty = (hits_border | target_survives).to(self.health.dtype)
        attacker_reward = target_killed.to(self.health.dtype) * KILL_REWARD
        new_health = (self.health - damage_received - attacker_penalty + attacker_reward).clamp_max(MAX_HEALTH)
        new_health = apply_movement_health_cost(new_health, hits_empty)

        self.flat_positions = torch.where(hits_empty | target_killed, target_flat_positions, self.flat_positions)
        stayed_put = active & (self.flat_positions == old_flat_positions)
        self.stationary_steps = torch.where(
            stayed_put,
            self.stationary_steps + 1,
            torch.zeros_like(self.stationary_steps),
        )
        self.health = apply_stationary_health_cost(new_health, self.stationary_steps, self.stationary_health_cap)
        self.stationary_steps = torch.where(
            self.health > 0,
            self.stationary_steps,
            torch.zeros_like(self.stationary_steps),
        )
        if sync_positions:
            self.sync_positions_from_flat()
        alive = self.health > 0
        self.update_grids_incremental(old_flat_positions, alive=alive)
        owns_position = self.index_grid.reshape(-1)[self.flat_positions] == self.index_grid_indices()
        consumed_food = alive & owns_position & hits_empty & (food_flat[target_flat_positions] > 0)
        self.health = torch.where(
            consumed_food,
            (self.health + torch.as_tensor(FOOD_REWARD, device=self.device, dtype=self.health.dtype)).clamp_max(MAX_HEALTH),
            self.health,
        )
        clear_consumed_food(self.food_grid, target_flat_positions, consumed_food)
        refresh_food_overlay(self.grid, self.food_grid)
        self.grid.reshape(-1)[self.flat_positions[alive]] = 1
        self.health = torch.where(
            alive & owns_position,
            self.health,
            torch.zeros_like(self.health),
        )
        self.stationary_steps = torch.where(
            self.health > 0,
            self.stationary_steps,
            torch.zeros_like(self.stationary_steps),
        )
        alive = self.health > 0
        if not compact_dead:
            return
        elif bool(alive.all()):
            return
        else:
            if not sync_positions:
                self.sync_positions_from_flat()
            self.compact(alive)

    def dense_weight_1(self):
        if self.cells == 0:
            return torch.empty(0, HIDDEN_DIM, INPUT_DIM, device=self.device)
        selected_base = self.base_weight_1[self.family_index]
        selected_direction = self.u_1.unsqueeze(2) * self.v_1.unsqueeze(1)
        return selected_base + self.coeff_1.reshape(-1, 1, 1) * selected_direction

    def dense_weight_2(self):
        if self.cells == 0:
            return torch.empty(0, OUTPUT_DIM, HIDDEN_DIM, device=self.device)
        selected_base = self.base_weight_2[self.family_index]
        selected_direction = self.u_2.unsqueeze(2) * self.v_2.unsqueeze(1)
        return selected_base + self.coeff_2.reshape(-1, 1, 1) * selected_direction

    def weighted_survivor_family(self):
        self.ensure_evolution_buffers()
        participants = self.round_participants
        if self.cells == 0 or not bool(participants.any()):
            base_weight_1, base_bias_1 = init_linear_weight_bias(HIDDEN_DIM, INPUT_DIM, 1, self.device)
            base_weight_2, base_bias_2 = init_linear_weight_bias(OUTPUT_DIM, HIDDEN_DIM, 1, self.device)
            base_weight_1 = base_weight_1[0]
            base_weight_2 = base_weight_2[0]
            base_bias_1 = base_bias_1[0]
            base_bias_2 = base_bias_2[0]
        else:
            fitness = (self.round_survival_steps[participants] / float(npd.ROUNDTIME)).clamp(0.0, 1.0)
            centered = fitness - fitness.mean()
            std = fitness.std(unbiased=False)
            if bool(std <= 1.0e-6):
                base_weight_1 = self.evolution_anchor_weight_1.clone()
                base_bias_1 = self.evolution_anchor_bias_1.clone()
                base_weight_2 = self.evolution_anchor_weight_2.clone()
                base_bias_2 = self.evolution_anchor_bias_2.clone()
            else:
                advantage = centered / std.clamp_min(1.0e-6)
                weight_1 = self.dense_weight_1()[participants]
                weight_2 = self.dense_weight_2()[participants]
                bias_1 = self.bias_1[participants]
                bias_2 = self.bias_2[participants]
                update_weight_1 = (
                    advantage.reshape(-1, 1, 1)
                    * (weight_1 - self.evolution_anchor_weight_1.unsqueeze(0))
                ).mean(dim=0)
                update_weight_2 = (
                    advantage.reshape(-1, 1, 1)
                    * (weight_2 - self.evolution_anchor_weight_2.unsqueeze(0))
                ).mean(dim=0)
                update_bias_1 = (
                    advantage.reshape(-1, 1)
                    * (bias_1 - self.evolution_anchor_bias_1.unsqueeze(0))
                ).mean(dim=0)
                update_bias_2 = (
                    advantage.reshape(-1, 1)
                    * (bias_2 - self.evolution_anchor_bias_2.unsqueeze(0))
                ).mean(dim=0)
                lr = float(npd.FITNESS_UPDATE_LR)
                base_weight_1 = self.evolution_anchor_weight_1 + lr * update_weight_1
                base_weight_2 = self.evolution_anchor_weight_2 + lr * update_weight_2
                base_bias_1 = self.evolution_anchor_bias_1 + lr * update_bias_1
                base_bias_2 = self.evolution_anchor_bias_2 + lr * update_bias_2

        self.evolution_anchor_weight_1 = base_weight_1.clone()
        self.evolution_anchor_bias_1 = base_bias_1.clone()
        self.evolution_anchor_weight_2 = base_weight_2.clone()
        self.evolution_anchor_bias_2 = base_bias_2.clone()

        return base_weight_1, base_bias_1, base_weight_2, base_bias_2

    def empty_positions(self):
        playable = self.index_grid[2:-2, 2:-2].reshape(-1)
        empty_flat = torch.nonzero(playable == -1, as_tuple=False).reshape(-1)
        if empty_flat.numel() == 0:
            return torch.empty(0, 2, device=self.device, dtype=torch.long)
        rows, cols = self.playable_shape
        positions = torch.empty(empty_flat.numel(), 2, device=self.device, dtype=torch.long)
        positions[:, 0] = empty_flat.div(cols, rounding_mode='floor') + 2
        positions[:, 1] = empty_flat.remainder(cols) + 2
        return positions

    def empty_flat_positions(self):
        playable = self.index_grid[2:-2, 2:-2].reshape(-1)
        empty_flat = torch.nonzero(playable == -1, as_tuple=False).reshape(-1)
        if empty_flat.numel() == 0:
            return torch.empty(0, device=self.device, dtype=torch.long)
        _rows, cols = self.playable_shape
        row = empty_flat.div(cols, rounding_mode='floor') + 2
        col = empty_flat.remainder(cols) + 2
        return row * self.grid_stride + col

    def append_weighted_wave(
            self,
            count,
            initial_health=2,
            coeff_scale=npd.FACTORED_WAVE_COEFF_SCALE,
            sync_existing_positions=True):
        count = int(count)
        if count <= 0:
            return 0
        if sync_existing_positions and self.cells > 0:
            self.sync_positions_from_flat()
        empties = self.empty_flat_positions()
        spawn_count = min(count, int(empties.shape[0]))
        if spawn_count <= 0:
            return 0

        old_cells = self.cells
        selection = torch.randperm(empties.shape[0], device=self.device)[:spawn_count]
        new_flat_positions = empties[selection]
        new_positions = torch.empty(spawn_count, 2, device=self.device, dtype=torch.long)
        new_positions[:, 0] = new_flat_positions.div(self.grid_stride, rounding_mode='floor')
        new_positions[:, 1] = new_flat_positions.remainder(self.grid_stride)
        (
            base_weight_1,
            base_bias_1,
            base_weight_2,
            base_bias_2,
        ) = self.weighted_survivor_family()

        new_family_id = self.families
        self.base_weight_1 = torch.cat((self.base_weight_1, base_weight_1.unsqueeze(0)), dim=0)
        self.base_weight_2 = torch.cat((self.base_weight_2, base_weight_2.unsqueeze(0)), dim=0)
        self.refresh_base_weight_matmul_cache()
        new_u_1 = torch.randn(spawn_count, HIDDEN_DIM, device=self.device)
        new_v_1 = torch.randn(spawn_count, INPUT_DIM, device=self.device)
        new_u_2 = torch.randn(spawn_count, OUTPUT_DIM, device=self.device)
        new_v_2 = torch.randn(spawn_count, HIDDEN_DIM, device=self.device)
        normalize_rank1_factor_rows_(new_u_1, new_v_1)
        normalize_rank1_factor_rows_(new_u_2, new_v_2)

        self.positions = torch.cat((self.positions, new_positions), dim=0)
        self.flat_positions = torch.cat((self.flat_positions, new_flat_positions), dim=0)
        self.health = torch.cat((
            self.health,
            torch.full((spawn_count,), initial_health, device=self.device, dtype=self.health.dtype),
        ), dim=0)
        self.stationary_steps = torch.cat((
            self.stationary_steps,
            torch.zeros(spawn_count, device=self.device, dtype=self.stationary_steps.dtype),
        ), dim=0)
        self.recurrent_state = torch.cat((
            self.recurrent_state,
            torch.zeros(spawn_count, HIDDEN_DIM, device=self.device),
        ), dim=0)
        self.family_index = torch.cat((
            self.family_index,
            torch.full((spawn_count,), new_family_id, device=self.device, dtype=self.family_index.dtype),
        ), dim=0)
        self.coeff_1 = torch.cat((self.coeff_1, torch.randn(spawn_count, device=self.device) * coeff_scale), dim=0)
        self.coeff_2 = torch.cat((self.coeff_2, torch.randn(spawn_count, device=self.device) * coeff_scale), dim=0)
        self.u_1 = torch.cat((self.u_1, new_u_1), dim=0)
        self.v_1 = torch.cat((self.v_1, new_v_1), dim=0)
        self.u_2 = torch.cat((self.u_2, new_u_2), dim=0)
        self.v_2 = torch.cat((self.v_2, new_v_2), dim=0)
        self.bias_1 = torch.cat((self.bias_1, base_bias_1.expand(spawn_count, -1).clone()), dim=0)
        self.bias_2 = torch.cat((self.bias_2, base_bias_2.expand(spawn_count, -1).clone()), dim=0)
        self.single_active_family_id = new_family_id if old_cells == 0 else None
        new_indices = torch.arange(
            old_cells,
            old_cells + spawn_count,
            device=self.device,
            dtype=self.index_grid.dtype,
        )
        grid_flat = self.grid.reshape(-1)
        index_flat = self.index_grid.reshape(-1)
        grid_flat[new_flat_positions] = 1
        index_flat[new_flat_positions] = new_indices
        self.reset_round_survival_tracking()
        return spawn_count

    def append_static_weighted_wave(
            self,
            family_count,
            count,
            initial_health=2,
            coeff_scale=npd.FACTORED_WAVE_COEFF_SCALE):
        count = int(count)
        family_count = int(family_count)
        if count <= 0:
            return 0, family_count
        inactive_slots = torch.nonzero(self.health <= 0, as_tuple=False).reshape(-1)
        empties = self.empty_flat_positions()
        spawn_count = min(count, int(inactive_slots.shape[0]), int(empties.shape[0]))
        if spawn_count <= 0:
            return 0, family_count
        new_family_id, family_count = self.next_static_family_slot(family_count)
        if new_family_id is None:
            return 0, family_count

        slot_selection = torch.randperm(inactive_slots.shape[0], device=self.device)[:spawn_count]
        empty_selection = torch.randperm(empties.shape[0], device=self.device)[:spawn_count]
        slots = inactive_slots[slot_selection]
        new_flat_positions = empties[empty_selection]
        (
            base_weight_1,
            base_bias_1,
            base_weight_2,
            base_bias_2,
        ) = self.weighted_survivor_family()

        self.base_weight_1[new_family_id] = base_weight_1
        self.base_weight_2[new_family_id] = base_weight_2
        self.refresh_base_weight_matmul_cache_row(new_family_id)
        new_u_1 = torch.randn(spawn_count, HIDDEN_DIM, device=self.device)
        new_v_1 = torch.randn(spawn_count, INPUT_DIM, device=self.device)
        new_u_2 = torch.randn(spawn_count, OUTPUT_DIM, device=self.device)
        new_v_2 = torch.randn(spawn_count, HIDDEN_DIM, device=self.device)
        normalize_rank1_factor_rows_(new_u_1, new_v_1)
        normalize_rank1_factor_rows_(new_u_2, new_v_2)

        self.flat_positions[slots] = new_flat_positions
        self.positions[slots, 0] = new_flat_positions.div(self.grid_stride, rounding_mode='floor')
        self.positions[slots, 1] = new_flat_positions.remainder(self.grid_stride)
        self.health[slots] = initial_health
        self.stationary_steps[slots] = 0
        self.recurrent_state[slots] = 0
        self.family_index[slots] = new_family_id
        self.coeff_1[slots] = torch.randn(spawn_count, device=self.device) * coeff_scale
        self.coeff_2[slots] = torch.randn(spawn_count, device=self.device) * coeff_scale
        self.u_1[slots] = new_u_1
        self.v_1[slots] = new_v_1
        self.u_2[slots] = new_u_2
        self.v_2[slots] = new_v_2
        self.bias_1[slots] = base_bias_1
        self.bias_2[slots] = base_bias_2

        grid_flat = self.grid.reshape(-1)
        index_flat = self.index_grid.reshape(-1)
        grid_flat[new_flat_positions] = 1
        index_flat[new_flat_positions] = slots.to(self.index_grid.dtype)
        self.single_active_family_id = None
        self.reset_round_survival_tracking()
        return spawn_count, family_count

    def compact(self, alive):
        self.sync_positions_from_flat()
        if bool(alive.all()):
            self.rebuild_grids()
            return
        self.positions = self.positions[alive]
        self.flat_positions = self.flat_positions[alive]
        self.health = self.health[alive]
        if self.round_survival_steps is not None and self.round_survival_steps.shape[0] == alive.shape[0]:
            self.round_survival_steps = self.round_survival_steps[alive]
        if self.round_participants is not None and self.round_participants.shape[0] == alive.shape[0]:
            self.round_participants = self.round_participants[alive]
        self.stationary_steps = self.stationary_steps[alive]
        self.recurrent_state = self.recurrent_state[alive]
        self.family_index = self.family_index[alive]
        self.coeff_1 = self.coeff_1[alive]
        self.coeff_2 = self.coeff_2[alive]
        self.u_1 = self.u_1[alive]
        self.v_1 = self.v_1[alive]
        self.u_2 = self.u_2[alive]
        self.v_2 = self.v_2[alive]
        self.bias_1 = self.bias_1[alive]
        self.bias_2 = self.bias_2[alive]
        self.prune_unused_families()
        self.rebuild_grids()

    def step(self, movement='none', compact_dead=True, sync_positions=True):
        if self.cells == 0:
            return torch.empty(0, device=self.device, dtype=torch.long)
        actions = self.forward_actions()
        if movement == 'snapshot':
            self.apply_snapshot_movement(actions, sync_positions=sync_positions)
        elif movement == 'snapshot_combat':
            self.apply_snapshot_combat(actions, compact_dead=compact_dead, sync_positions=sync_positions)
        return actions

    def compiled_snapshot_combat_step(self, rebuild_grid=False, family_basis=False, compile_mode='reduce-overhead'):
        if family_basis:
            step_fn = compiled_family_basis_rebuild_snapshot_combat_step_tensors(compile_mode)
            (
                self.flat_positions,
                self.health,
                self.stationary_steps,
                self.recurrent_state,
                actions,
                self.round_survival_steps,
                event_counts,
            ) = step_fn(*self.family_basis_block_args())
            return event_counts
        else:
            if rebuild_grid:
                step_fn = compiled_rebuild_snapshot_combat_step_tensors(compile_mode)
            else:
                step_fn = compiled_snapshot_combat_step_tensors(compile_mode)
            self.flat_positions, self.health, self.recurrent_state, actions = step_fn(
                self.grid,
                self.index_grid,
                self.food_grid,
                self.flat_positions,
                self.health,
                self.recurrent_state,
                self.family_index,
                self.coeff_1,
                self.coeff_2,
                self.bias_1,
                self.bias_2,
                self.base_weight_1,
                self.base_weight_2,
                self.u_1,
                self.v_1,
                self.u_2,
                self.v_2,
                self.neighbor_flat_offsets,
                self.direction_flat_deltas,
            )
        return actions

    def compiled_snapshot_combat_steps(self, block_steps, rebuild_grid=False, family_basis=False, compile_mode='default'):
        block_steps = int(block_steps)
        if block_steps == 1:
            return self.compiled_snapshot_combat_step(
                rebuild_grid=rebuild_grid,
                family_basis=family_basis,
                compile_mode=compile_mode,
            )
        if not (rebuild_grid and family_basis):
            raise ValueError('compiled block steps currently require rebuild_grid and family_basis')
        step_fn = compiled_family_basis_rebuild_snapshot_combat_block_tensors(block_steps, compile_mode)
        (
            self.flat_positions,
            self.health,
            self.stationary_steps,
            self.recurrent_state,
            self.round_survival_steps,
            event_counts,
        ) = step_fn(*self.family_basis_block_args())
        return event_counts

def benchmark_tensor_state(
        cells,
        height,
        width,
        families,
        steps,
        warmup_steps,
        movement,
        device,
        initial_health=2,
        wave_every=0,
        wave_size=0,
        wave_initial_health=2,
        compact_every=1,
        checksum_actions=1024,
        trace_every=0,
        compiled_step=False,
        static_capacity=False,
        family_capacity=None,
        cell_capacity=None,
        static_refill_empty=False,
        static_refill_check_every=1,
        health_dtype='float32',
        coeff_scale=npd.FACTORED_WAVE_COEFF_SCALE,
        stationary_health_cap=1,
        static_rebuild_grid=False,
        family_basis_step=False,
        matmul_precision=None,
        compile_mode='reduce-overhead',
        compiled_block_steps=1,
        cuda_graph_block=False,
        normal_round_refill=False,
        early_end_empty_round=False,
        per_wave=None,
        min_wave=None):
    compiled_step = bool(compiled_step)
    static_capacity = bool(static_capacity)
    static_refill_empty = bool(static_refill_empty)
    static_rebuild_grid = bool(static_rebuild_grid)
    family_basis_step = bool(family_basis_step)
    cuda_graph_block = bool(cuda_graph_block)
    normal_round_refill = bool(normal_round_refill)
    early_end_empty_round = bool(early_end_empty_round)
    static_refill_check_every = max(int(static_refill_check_every), 1)
    compiled_block_steps = max(int(compiled_block_steps), 1)
    per_wave = npd.PER_WAVE if per_wave is None else int(per_wave)
    min_wave = npd.MIN_WAVE if min_wave is None else int(min_wave)
    health_dtype_name = health_dtype
    health_dtype = resolve_health_dtype(health_dtype)
    compile_mode = resolve_compile_mode(compile_mode)
    if matmul_precision is not None:
        if matmul_precision not in MATMUL_PRECISIONS:
            raise ValueError(f'unsupported matmul precision: {matmul_precision}')
        torch.set_float32_matmul_precision(matmul_precision)
    if static_refill_empty and not static_capacity:
        raise ValueError('static_refill_empty requires static_capacity')
    if static_refill_empty and family_capacity is None:
        raise ValueError('static_refill_empty requires explicit family_capacity')
    if static_rebuild_grid and not static_capacity:
        raise ValueError('static_rebuild_grid requires static_capacity')
    if static_rebuild_grid and not compiled_step:
        raise ValueError('static_rebuild_grid requires compiled_step')
    if family_basis_step and not static_rebuild_grid:
        raise ValueError('family_basis_step requires static_rebuild_grid')
    if static_capacity:
        if movement != 'snapshot_combat':
            raise ValueError('static_capacity currently supports movement="snapshot_combat" only')
        if compact_every != 0:
            raise ValueError('static_capacity requires compact_every=0')
        if family_capacity is None:
            scheduled_waves = steps // wave_every if wave_every > 0 else 0
            family_capacity = families + scheduled_waves + 1
        family_capacity = int(family_capacity)
        if family_capacity < families:
            raise ValueError('family_capacity must be at least families')
        if cell_capacity is not None:
            cell_capacity = int(cell_capacity)
            if cell_capacity < cells:
                raise ValueError('cell_capacity must be at least cells')
    if compiled_step:
        if torch.device(device).type != 'cuda':
            raise ValueError('compiled_step requires a CUDA device')
        if movement != 'snapshot_combat':
            raise ValueError('compiled_step currently supports movement=\"snapshot_combat\" only')
        if compact_every != 0:
            raise ValueError('compiled_step requires compact_every=0 to keep tensor shapes stable')
        if not static_capacity and (wave_every != 0 or wave_size != 0):
            raise ValueError('compiled_step requires wave_every=0 and wave_size=0 unless static_capacity is enabled')
    if compiled_block_steps > 1:
        if not compiled_step:
            raise ValueError('compiled_block_steps requires compiled_step')
        if not static_capacity:
            raise ValueError('compiled_block_steps requires static_capacity')
        if not (static_rebuild_grid and family_basis_step):
            raise ValueError('compiled_block_steps requires static_rebuild_grid and family_basis_step')
        if checksum_actions:
            raise ValueError('compiled_block_steps requires checksum_actions=0')
    if cuda_graph_block:
        if not (compiled_step and compiled_block_steps > 1):
            raise ValueError('cuda_graph_block requires compiled block steps')
        if not (static_capacity and static_rebuild_grid and family_basis_step):
            raise ValueError('cuda_graph_block requires static rebuild-grid family-basis mode')
        if torch.device(device).type != 'cuda':
            raise ValueError('cuda_graph_block requires a CUDA device')
    if normal_round_refill:
        if per_wave <= 0:
            raise ValueError('normal_round_refill requires per_wave > 0')
        if min_wave <= 0:
            raise ValueError('normal_round_refill requires min_wave > 0')
    if static_capacity:
        state = TensorRank1State.fixed_capacity(
            active_cells=cells,
            height=height,
            width=width,
            active_families=families,
            family_capacity=family_capacity,
            device=device,
            initial_health=initial_health,
            cell_capacity=cell_capacity,
            health_dtype=health_dtype,
            coeff_scale=coeff_scale,
            stationary_health_cap=stationary_health_cap,
        )
        active_family_count = families
    else:
        state = TensorRank1State.random(
            cells=cells,
            height=height,
            width=width,
            families=families,
            device=device,
            initial_health=initial_health,
            health_dtype=health_dtype,
            coeff_scale=coeff_scale,
            stationary_health_cap=stationary_health_cap,
        )
        active_family_count = state.families
    sync_positions_each_step = False
    checksum_actions = max(int(checksum_actions), 0)
    checksum_tensor = torch.zeros((), device=state.device, dtype=torch.long) if checksum_actions else None
    processed_cell_steps = 0
    waves_spawned = 0
    empty_refills = 0
    trace_every = max(int(trace_every), 0)
    trace_segments = []
    segment_start = None
    segment_processed_cell_steps = 0
    segment_waves_spawned = 0
    segment_empty_refills = 0
    segment_start_cells = 0
    segment_start_active_cells = 0
    segment_start_families = 0
    completed_steps = 0
    graph_block_runners = {}
    timed_cuda_graph_captures = 0
    cuda_graph_replay_enabled = False
    last_active_cells = None

    def refresh_active_cell_count():
        nonlocal last_active_cells
        last_active_cells = int((state.health > 0).sum().item())
        return last_active_cells

    def active_cell_count():
        if last_active_cells is not None:
            return last_active_cells
        return refresh_active_cell_count()

    def apply_known_spawn_count(spawned):
        nonlocal last_active_cells
        if last_active_cells is not None:
            last_active_cells += int(spawned)

    def clear_cuda_graphs_if_family_capacity_changed(previous_version):
        if state.family_capacity_version() != previous_version:
            graph_block_runners.clear()

    def invalidate_active_cell_count():
        nonlocal last_active_cells
        last_active_cells = None

    def active_cell_count_for_refill_check():
        return active_cell_count()

    def final_active_cell_count():
        return int((state.health > 0).sum().item())

    def scheduled_wave_size():
        if normal_round_refill:
            return max(per_wave - active_cell_count(), min_wave)
        return wave_size

    def empty_refill_size():
        if normal_round_refill:
            return min_wave
        return wave_size

    def append_trace_segment(start_step, end_step, segment_seconds, end_active_cells):
        trace_segments.append({
            'start_step': start_step,
            'end_step': end_step,
            'seconds': segment_seconds,
            'processed_cell_steps': segment_processed_cell_steps,
            'cells_per_second': segment_processed_cell_steps / segment_seconds if segment_seconds > 0 else 0.0,
            'cells_start': segment_start_cells,
            'cells_end': state.cells,
            'active_cells_start': segment_start_active_cells,
            'active_cells_end': end_active_cells,
            'families_start': segment_start_families,
            'families_end': state.families,
            'waves_spawned': segment_waves_spawned,
            'waves_spawned_total': waves_spawned,
            'empty_refills': segment_empty_refills,
            'empty_refills_total': empty_refills,
        })

    def run_compiled_steps(step_count):
        nonlocal timed_cuda_graph_captures
        if cuda_graph_block and cuda_graph_replay_enabled:
            runner = graph_block_runners.get(step_count)
            if runner is None:
                compile_state = state.clone()
                compile_state.compiled_snapshot_combat_steps(
                    step_count,
                    rebuild_grid=static_rebuild_grid,
                    family_basis=family_basis_step,
                    compile_mode=compile_mode,
                )
                synchronize(state.device)
                runner = CudaGraphFamilyBasisBlockRunner(
                    state,
                    step_count,
                    compile_mode,
                )
                graph_block_runners[step_count] = runner
                timed_cuda_graph_captures += 1
            runner.replay()
            return None
        if step_count == 1:
            return state.compiled_snapshot_combat_step(
                rebuild_grid=static_rebuild_grid,
                family_basis=family_basis_step,
                compile_mode=compile_mode,
            )
        return state.compiled_snapshot_combat_steps(
            step_count,
            rebuild_grid=static_rebuild_grid,
            family_basis=family_basis_step,
            compile_mode=compile_mode,
        )

    def next_block_step_count(step_index):
        step_count = min(compiled_block_steps, steps - step_index)
        if static_refill_empty:
            step_count = min(step_count, static_refill_check_every - (step_index % static_refill_check_every))
        if wave_every > 0:
            step_count = min(step_count, wave_every - (step_index % wave_every))
        if trace_every:
            step_count = min(step_count, trace_every - (step_index % trace_every))
        return max(step_count, 1)

    def compiled_block_counts_to_prewarm():
        counts = set()
        step_index = 0
        while step_index < steps:
            step_count = next_block_step_count(step_index)
            counts.add(step_count)
            step_index += step_count
        return counts

    grad_context = torch.no_grad() if compiled_step else torch.inference_mode()
    with grad_context:
        cuda_graph_replay_enabled = False
        if compiled_step and compiled_block_steps > 1:
            benchmark_state = state
            state = benchmark_state.clone()
            for step_count in sorted(compiled_block_counts_to_prewarm()):
                run_compiled_steps(step_count)
            synchronize(state.device)
            state = benchmark_state
            if cuda_graph_block:
                for step_count in sorted(compiled_block_counts_to_prewarm()):
                    graph_block_runners[step_count] = CudaGraphFamilyBasisBlockRunner(
                        state,
                        step_count,
                        compile_mode,
                    )
                synchronize(state.device)
            cuda_graph_replay_enabled = True
        warmup_step = 0
        while warmup_step < max(warmup_steps, 0):
            warmup_count = min(compiled_block_steps, max(warmup_steps, 0) - warmup_step)
            if compiled_step:
                run_compiled_steps(warmup_count)
            else:
                state.step(
                    movement=movement,
                    compact_dead=(compact_every == 1),
                    sync_positions=sync_positions_each_step,
                )
            warmup_step += warmup_count
            if compact_every > 1 and warmup_step % compact_every == 0:
                state.compact(state.alive_mask())
        synchronize(state.device)
        started = time.perf_counter()
        if trace_every:
            segment_start_cells = state.cells
            segment_start_active_cells = refresh_active_cell_count()
            segment_start_families = state.families
            segment_start = time.perf_counter()
        _step = 0
        while _step < steps:
            if static_refill_empty and _step % static_refill_check_every == 0:
                if active_cell_count_for_refill_check() == 0:
                    previous_family_version = state.family_capacity_version()
                    spawned, active_family_count = state.append_static_weighted_wave(
                        active_family_count,
                        empty_refill_size(),
                        initial_health=wave_initial_health,
                        coeff_scale=coeff_scale,
                    )
                    clear_cuda_graphs_if_family_capacity_changed(previous_family_version)
                    apply_known_spawn_count(spawned)
                    waves_spawned += spawned
                    segment_waves_spawned += spawned
                    empty_refills += 1
                    segment_empty_refills += 1
                    if spawned == 0:
                        break
            if not static_capacity and state.cells == 0:
                if empty_refill_size() <= 0:
                    break
                spawned = state.append_weighted_wave(
                    empty_refill_size(),
                    initial_health=wave_initial_health,
                    sync_existing_positions=False,
                    coeff_scale=coeff_scale,
                )
                apply_known_spawn_count(spawned)
                waves_spawned += spawned
                segment_waves_spawned += spawned
                empty_refills += 1
                segment_empty_refills += 1
                if spawned == 0:
                    break
            step_count = next_block_step_count(_step) if compiled_step else 1
            processed_cell_steps += state.cells * step_count
            segment_processed_cell_steps += state.cells * step_count
            if compiled_step:
                actions = run_compiled_steps(step_count)
            else:
                actions = state.step(
                    movement=movement,
                    compact_dead=(compact_every == 1),
                    sync_positions=sync_positions_each_step,
                )
            if not (compiled_step and family_basis_step):
                state.record_survival_steps(step_count)
            invalidate_active_cell_count()
            if checksum_tensor is not None:
                checksum_tensor = checksum_tensor + actions[:checksum_actions].sum()
            compacted_this_step = False
            _step += step_count
            if compact_every > 1 and _step % compact_every == 0:
                state.compact(state.alive_mask())
                compacted_this_step = True
            if (
                    early_end_empty_round
                    and wave_every > 0
                    and _step % wave_every != 0
                    and active_cell_count_for_refill_check() == 0):
                _step = min(steps, _step + (wave_every - (_step % wave_every)))
            if wave_every > 0 and _step % wave_every == 0:
                state.apply_round_transition_health_cost(compact_dead=(not static_capacity and compact_every == 1))
                invalidate_active_cell_count()
                round_wave_size = scheduled_wave_size()
                if static_capacity:
                    previous_family_version = state.family_capacity_version()
                    spawned, active_family_count = state.append_static_weighted_wave(
                        active_family_count,
                        round_wave_size,
                        initial_health=wave_initial_health,
                        coeff_scale=coeff_scale,
                    )
                    clear_cuda_graphs_if_family_capacity_changed(previous_family_version)
                    apply_known_spawn_count(spawned)
                    waves_spawned += spawned
                    segment_waves_spawned += spawned
                elif compact_every != 1 and not compacted_this_step:
                    state.compact(state.alive_mask())
                    invalidate_active_cell_count()
                    spawned = state.append_weighted_wave(
                        round_wave_size,
                        initial_health=wave_initial_health,
                        sync_existing_positions=(compact_every == 1),
                        coeff_scale=coeff_scale,
                    )
                    apply_known_spawn_count(spawned)
                    waves_spawned += spawned
                    segment_waves_spawned += spawned
                else:
                    spawned = state.append_weighted_wave(
                        round_wave_size,
                        initial_health=wave_initial_health,
                        sync_existing_positions=(compact_every == 1),
                        coeff_scale=coeff_scale,
                    )
                    apply_known_spawn_count(spawned)
                    waves_spawned += spawned
                    segment_waves_spawned += spawned
            if trace_every and _step % trace_every == 0:
                synchronize(state.device)
                now = time.perf_counter()
                end_active_cells = active_cell_count()
                append_trace_segment(_step - trace_every, _step, now - segment_start, end_active_cells)
                segment_processed_cell_steps = 0
                segment_waves_spawned = 0
                segment_empty_refills = 0
                segment_start_cells = state.cells
                segment_start_active_cells = end_active_cells
                segment_start_families = state.families
                segment_start = time.perf_counter()
            completed_steps = _step
        synchronize(state.device)
        if trace_every and segment_processed_cell_steps:
            now = time.perf_counter()
            end_active_cells = active_cell_count()
            start_step = trace_segments[-1]['end_step'] if trace_segments else 0
            append_trace_segment(start_step, completed_steps, now - segment_start, end_active_cells)
    checksum = int(checksum_tensor.item()) if checksum_tensor is not None else None
    elapsed = time.perf_counter() - started
    active_cells_final = active_cell_count() if last_active_cells is not None else final_active_cell_count()
    metrics = {
        'action_checksum': checksum,
        'cells': cells,
        'cell_capacity': state.cells if static_capacity else None,
        'cells_final': state.cells,
        'active_cells_final': active_cells_final,
        'cells_per_second': processed_cell_steps / elapsed,
        'compact_every': compact_every,
        'checksum_actions': checksum_actions,
        'compiled_step': compiled_step,
        'compiled_block_steps': compiled_block_steps if compiled_step else None,
        'compile_mode': compile_mode if compiled_step else None,
        'coeff_scale': coeff_scale,
        'stationary_health_cap': stationary_health_cap,
        'cuda_graph_block': cuda_graph_block,
        'timed_cuda_graph_captures': timed_cuda_graph_captures if cuda_graph_block else None,
        'cuda_name': torch.cuda.get_device_name(state.device) if state.device.type == 'cuda' else '',
        'device': str(state.device),
        'elapsed_seconds': elapsed,
        'empty_refills': empty_refills,
        'early_end_empty_round': early_end_empty_round,
        'families': families,
        'families_final': state.families,
        'active_families_final': active_family_count if static_capacity else state.families,
        'family_capacity': family_capacity,
        'family_basis_step': family_basis_step,
        'fitness_update_lr': npd.FITNESS_UPDATE_LR,
        'height': height,
        'health_dtype': health_dtype_name,
        'initial_health': initial_health,
        'movement': movement,
        'min_wave': min_wave if normal_round_refill else None,
        'normal_round_refill': normal_round_refill,
        'matmul_precision': matmul_precision,
        'per_wave': per_wave if normal_round_refill else None,
        'processed_cell_steps': processed_cell_steps,
        'steps': steps,
        'static_capacity': static_capacity,
        'static_rebuild_grid': static_rebuild_grid,
        'static_refill_check_every': static_refill_check_every if static_refill_empty else None,
        'static_refill_empty': static_refill_empty,
        'wave_every': wave_every,
        'wave_initial_health': wave_initial_health,
        'wave_size': wave_size,
        'warmup_steps': max(warmup_steps, 0),
        'waves_spawned': waves_spawned,
        'width': width,
    }
    if trace_every:
        metrics['trace_every'] = trace_every
        metrics['trace_segments'] = trace_segments
    return metrics
