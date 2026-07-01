#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import neural_petri_dish as npd
from tensor_rank1_sim import KILL_REWARD, MAX_HEALTH, TensorRank1State, resolve_device, resolve_health_dtype, synchronize


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('value must be positive')
    return parsed


def nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError('value must be non-negative')
    return parsed


def coeff_scale_list(value):
    parsed = [float(item.strip()) for item in value.split(',') if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError('expected at least one coefficient scale')
    if any(scale < 0 for scale in parsed):
        raise argparse.ArgumentTypeError('coefficient scales must be non-negative')
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure border-hit and edge-death rates for tensor rank-1 normal-play runs.'
    )
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--size', type=npd.parse_size, default=(60, 80))
    parser.add_argument('--initial-cells', type=nonnegative_int, default=2500)
    parser.add_argument('--rounds', type=positive_int, default=200)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--coeff-scales', type=coeff_scale_list, default=[0.0003, 0.001, 0.003, 0.01, 0.03])
    parser.add_argument('--families', type=positive_int, default=1)
    parser.add_argument('--family-capacity', type=positive_int, default=10)
    parser.add_argument('--cell-capacity', type=positive_int)
    parser.add_argument('--initial-health', type=positive_int, default=15)
    parser.add_argument('--wave-initial-health', type=positive_int, default=2)
    parser.add_argument('--stationary-health-cap', type=int, default=1)
    parser.add_argument('--health-dtype', choices=('int64', 'int32'), default='int32')
    parser.add_argument('--output-json')
    parser.add_argument('--output-csv')
    return parser.parse_args()


def add_counts(total, counts):
    for key, value in counts.items():
        total[key] = total.get(key, 0) + int(value)


def count_tensor(mask):
    return int(mask.sum().item())


def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def action_counts(state, actions):
    health = state.health
    active = health > 0
    moving = active & (actions != 0)
    target_flat_positions = state.flat_positions + state.direction_flat_deltas[actions]
    target_indices = state.index_grid.reshape(-1)[target_flat_positions]

    hits_border = moving & (target_indices == -2)
    hits_empty = moving & (target_indices == -1)
    hits_occupied = moving & (target_indices >= 0)

    valid_targets = target_indices.clamp_min(0).to(torch.long)
    damage_received = torch.zeros_like(health)
    damage_received.scatter_add_(0, valid_targets, hits_occupied.to(health.dtype))
    target_health_after = health[valid_targets] - damage_received[valid_targets]
    target_survives = hits_occupied & (target_health_after > 0)
    target_killed = hits_occupied & (target_health_after <= 0)
    predicted_health = (
        health
        - damage_received
        - (hits_border | target_survives).to(health.dtype)
        + target_killed.to(health.dtype) * KILL_REWARD
    ).clamp_max(MAX_HEALTH)

    rows = state.flat_positions.div(state.grid_stride, rounding_mode='floor')
    cols = state.flat_positions.remainder(state.grid_stride)
    edge_cells = active & (
        (rows == 2)
        | (rows == state.grid.shape[0] - 3)
        | (cols == 2)
        | (cols == state.grid.shape[1] - 3)
    )

    return {
        'active_observations': count_tensor(active),
        'edge_cell_observations': count_tensor(edge_cells),
        'stationary_actions': count_tensor(active & (actions == 0)),
        'move_intents': count_tensor(moving),
        'empty_move_intents': count_tensor(hits_empty),
        'occupied_move_intents': count_tensor(hits_occupied),
        'border_hit_intents': count_tensor(hits_border),
        'edge_cell_border_hit_intents': count_tensor(edge_cells & hits_border),
        'target_survives_intents': count_tensor(target_survives),
        'kill_intents': count_tensor(target_killed),
        'predicted_border_deaths': count_tensor(hits_border & (predicted_health <= 0)),
    }


def summarize_counts(counts, steps, elapsed_seconds, state, active_family_count, waves_spawned, coeff_scale):
    active_observations = counts.get('active_observations', 0)
    move_intents = counts.get('move_intents', 0)
    occupied_move_intents = counts.get('occupied_move_intents', 0)
    edge_cell_observations = counts.get('edge_cell_observations', 0)
    border_hit_intents = counts.get('border_hit_intents', 0)
    total_deaths = counts.get('total_deaths', 0)
    summary = {
        **counts,
        'coeff_scale': coeff_scale,
        'steps': steps,
        'rounds': steps // npd.ROUNDTIME,
        'elapsed_seconds': elapsed_seconds,
        'active_cells_final': int((state.health > 0).sum().item()),
        'active_cells_mean': safe_div(active_observations, steps),
        'cell_capacity': state.cells,
        'families_final': state.families,
        'active_families_final': active_family_count,
        'waves_spawned': waves_spawned,
        'stationary_rate': safe_div(counts.get('stationary_actions', 0), active_observations),
        'move_intent_rate': safe_div(move_intents, active_observations),
        'border_hit_rate_per_alive': safe_div(border_hit_intents, active_observations),
        'border_hit_rate_per_move': safe_div(border_hit_intents, move_intents),
        'edge_cell_border_hit_rate': safe_div(counts.get('edge_cell_border_hit_intents', 0), edge_cell_observations),
        'empty_move_rate_per_move': safe_div(counts.get('empty_move_intents', 0), move_intents),
        'occupied_move_rate_per_move': safe_div(occupied_move_intents, move_intents),
        'kill_intent_rate_per_occupied': safe_div(counts.get('kill_intents', 0), occupied_move_intents),
        'death_rate_per_alive': safe_div(total_deaths, active_observations),
        'predicted_border_death_rate_per_alive': safe_div(counts.get('predicted_border_deaths', 0), active_observations),
    }
    return summary


def run_case(args, coeff_scale):
    size = npd.terminal_size(args.size)
    device = resolve_device(args.device)
    health_dtype = resolve_health_dtype(args.health_dtype)
    torch.manual_seed(args.seed)
    state = TensorRank1State.fixed_capacity(
        active_cells=args.initial_cells,
        height=size.lines,
        width=size.columns,
        active_families=args.families,
        family_capacity=args.family_capacity,
        device=device,
        initial_health=args.initial_health,
        cell_capacity=args.cell_capacity,
        health_dtype=health_dtype,
        coeff_scale=coeff_scale,
        stationary_health_cap=args.stationary_health_cap,
    )
    active_family_count = args.families
    counts = {}
    waves_spawned = 0
    total_steps = args.rounds * npd.ROUNDTIME
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for step_index in range(total_steps):
            alive_before = state.health > 0
            actions = state.forward_actions()
            add_counts(counts, action_counts(state, actions))
            state.apply_snapshot_combat(actions, compact_dead=False, sync_positions=False)
            died = alive_before & (state.health <= 0)
            counts['total_deaths'] = counts.get('total_deaths', 0) + count_tensor(died)

            if (step_index + 1) % npd.ROUNDTIME == 0:
                state.apply_round_transition_health_cost()
                active_cells = int((state.health > 0).sum().item())
                wave_size = max(npd.PER_WAVE - active_cells, npd.MIN_WAVE)
                spawned, active_family_count = state.append_static_weighted_wave(
                    active_family_count,
                    wave_size,
                    initial_health=args.wave_initial_health,
                    coeff_scale=coeff_scale,
                )
                waves_spawned += spawned
    synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    return summarize_counts(counts, total_steps, elapsed_seconds, state, active_family_count, waves_spawned, coeff_scale)


def write_csv(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with output.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows = []
    for coeff_scale in args.coeff_scales:
        summary = run_case(args, coeff_scale)
        rows.append(summary)
        print(json.dumps(summary, sort_keys=True))

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if args.output_csv:
        write_csv(args.output_csv, rows)


if __name__ == '__main__':
    main()
