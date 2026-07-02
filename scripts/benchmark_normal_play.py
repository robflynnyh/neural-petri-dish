#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import neural_petri_dish as npd
from tensor_rank1_sim import (
    COMPILE_MODES,
    EVENT_COUNT_NAMES,
    HEALTH_DTYPES,
    MATMUL_PRECISIONS,
    NETWORK_DTYPES,
    benchmark_tensor_state,
    resolve_device,
)


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('value must be positive')
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark normal Neural Petri Dish round time.')
    parser.add_argument('--engine', choices=('game', 'tensor_rank1'), default='game')
    parser.add_argument('--size', type=npd.parse_size, default=(60, 80))
    parser.add_argument('--initial-cells', type=positive_int, default=2500)
    parser.add_argument('--rounds', type=positive_int, default=3)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--action-backend', choices=npd.ACTION_BACKENDS, default=npd.ACTION_BACKEND_SEQUENTIAL)
    parser.add_argument('--action-device', choices=('auto', 'cpu', 'cuda'), default='cpu')
    parser.add_argument('--batched-min-family-size', type=positive_int, default=4096)
    parser.add_argument('--tensor-initial-health', type=positive_int, default=15)
    parser.add_argument('--tensor-wave-initial-health', type=positive_int, default=2)
    parser.add_argument('--tensor-coeff-scale', type=float, default=npd.FACTORED_WAVE_COEFF_SCALE)
    parser.add_argument('--fitness-update-lr', type=float, default=npd.FITNESS_UPDATE_LR)
    parser.add_argument('--tensor-stationary-health-cap', type=int, default=1)
    parser.add_argument('--tensor-family-capacity', type=positive_int, default=10)
    parser.add_argument('--tensor-cell-capacity', type=positive_int)
    parser.add_argument('--npc-count', type=int, default=npd.NPC_COUNT)
    parser.add_argument('--tensor-static-refill-check-every', type=positive_int, default=100)
    parser.add_argument('--tensor-health-dtype', choices=tuple(HEALTH_DTYPES), default='float32')
    parser.add_argument('--tensor-network-dtype', choices=('auto', *tuple(NETWORK_DTYPES)), default='auto')
    parser.add_argument('--tensor-matmul-precision', choices=MATMUL_PRECISIONS, default='high')
    parser.add_argument('--tensor-compile-mode', choices=COMPILE_MODES, default='default')
    parser.add_argument('--tensor-compiled-block-steps', type=positive_int, default=10)
    parser.add_argument('--tensor-cuda-graph-block', dest='no_tensor_cuda_graph_block', action='store_false')
    parser.add_argument('--no-tensor-cuda-graph-block', dest='no_tensor_cuda_graph_block', action='store_true')
    parser.set_defaults(no_tensor_cuda_graph_block=True)
    parser.add_argument('--collect-event-counts', action='store_true')
    parser.add_argument('--include-round-event-counts', action='store_true')
    parser.add_argument('--quiet', action='store_true', help='suppress stdout JSON; useful with --output-json')
    parser.add_argument('--output-json')
    args = parser.parse_args()
    if args.engine == 'tensor_rank1' and args.action_device == 'cpu':
        parser.error('--engine tensor_rank1 requires --action-device cuda or --action-device auto')
    return args


def event_window_summary(trace_segments, start, end):
    rows = trace_segments[start:end]
    if not rows:
        return {}
    counts = {name: 0 for name in EVENT_COUNT_NAMES}
    for row in rows:
        event_counts = row.get('event_counts', {})
        for name in EVENT_COUNT_NAMES:
            counts[name] += int(event_counts.get(name, 0))
    active_steps = max(1, counts['active_cell_steps'])
    visible_steps = max(1, counts['npc_visible_cell_steps'])
    counts.update({
        'round_start': int(start + 1),
        'round_end': int(end),
        'rounds': int(len(rows)),
        'seconds': float(sum(row['seconds'] for row in rows)),
        'move_success_rate': counts['move_successes'] / active_steps,
        'death_rate': counts['deaths'] / active_steps,
        'npc_kill_rate': counts['npc_kills'] / active_steps,
        'npc_adjacent_fraction': counts['npc_adjacent_cell_steps'] / active_steps,
        'npc_visible_death_rate': counts['npc_visible_deaths'] / visible_steps,
        'npc_visible_move_away_rate': counts['npc_visible_move_away'] / visible_steps,
        'npc_visible_move_toward_rate': counts['npc_visible_move_toward'] / visible_steps,
        'npc_visible_move_same_rate': counts['npc_visible_move_same'] / visible_steps,
        'npc_visible_final_alive_rate': counts['npc_visible_final_alive'] / visible_steps,
        'npc_visible_final_clear_rate': counts['npc_visible_final_clear'] / visible_steps,
        'npc_visible_final_adjacent_rate': counts['npc_visible_final_adjacent'] / visible_steps,
        'npc_visible_final_farther_rate': counts['npc_visible_final_farther'] / visible_steps,
        'npc_visible_final_closer_rate': counts['npc_visible_final_closer'] / visible_steps,
        'stayed_put_rate': counts['stayed_put'] / active_steps,
    })
    return counts


def add_event_count_summary(metrics, window=100):
    trace_segments = metrics.get('trace_segments', [])
    if not trace_segments or 'event_counts' not in trace_segments[0]:
        return
    window = min(int(window), len(trace_segments))
    metrics['event_count_summary'] = {
        'window_rounds': window,
        'all': event_window_summary(trace_segments, 0, len(trace_segments)),
        'first_window': event_window_summary(trace_segments, 0, window),
        'last_window': event_window_summary(trace_segments, len(trace_segments) - window, len(trace_segments)),
    }


def strip_round_event_counts(metrics):
    for segment in metrics.get('trace_segments', []):
        segment.pop('event_counts', None)


def run_tensor_rank1(args):
    npd.FITNESS_UPDATE_LR = args.fitness_update_lr
    torch.manual_seed(args.seed)
    size = npd.terminal_size(args.size)
    device_name = args.action_device
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics = benchmark_tensor_state(
        cells=args.initial_cells,
        height=size.lines,
        width=size.columns,
        families=1,
        steps=args.rounds * npd.ROUNDTIME,
        warmup_steps=0,
        movement='snapshot_combat',
        device=resolve_device(device_name),
        initial_health=args.tensor_initial_health,
        health_dtype=args.tensor_health_dtype,
        network_dtype=args.tensor_network_dtype,
        coeff_scale=args.tensor_coeff_scale,
        stationary_health_cap=args.tensor_stationary_health_cap,
        wave_every=npd.ROUNDTIME,
        wave_size=npd.PER_WAVE,
        wave_initial_health=args.tensor_wave_initial_health,
        compact_every=0,
        checksum_actions=0,
        trace_every=npd.ROUNDTIME,
        compiled_step=True,
        static_capacity=True,
        family_capacity=args.tensor_family_capacity,
        cell_capacity=args.tensor_cell_capacity,
        static_refill_empty=False,
        static_refill_check_every=args.tensor_static_refill_check_every,
        static_rebuild_grid=True,
        family_basis_step=True,
        matmul_precision=args.tensor_matmul_precision,
        compile_mode=args.tensor_compile_mode,
        compiled_block_steps=args.tensor_compiled_block_steps,
        cuda_graph_block=not args.no_tensor_cuda_graph_block,
        normal_round_refill=True,
        early_end_empty_round=True,
        per_wave=npd.PER_WAVE,
        min_wave=npd.MIN_WAVE,
        npc_count=args.npc_count,
        collect_event_counts=args.collect_event_counts,
    )
    if args.collect_event_counts:
        add_event_count_summary(metrics)
    completed = []
    for index, segment in enumerate(metrics.get('trace_segments', []), start=1):
        round_metrics = {
            'round': index,
            'seconds': segment['seconds'],
            'cells': segment['active_cells_end'],
            'waves_spawned': segment['waves_spawned'],
            'empty_refills': segment['empty_refills'],
        }
        if args.include_round_event_counts and 'event_counts' in segment:
            round_metrics['event_counts'] = segment['event_counts']
        completed.append(round_metrics)
    if args.collect_event_counts and not args.include_round_event_counts:
        strip_round_event_counts(metrics)
    return {
        'engine': args.engine,
        'semantic_note': (
            'GPU-resident tensor rank-1 engine candidate for normal-size rounds; '
            'interactive Game/Cell rendering is still the game engine path.'
        ),
        'action_backend': args.action_backend,
        'action_device': device_name,
        'batched_min_family_size': args.batched_min_family_size,
        'cuda_name': metrics['cuda_name'],
        'initial_cells': args.initial_cells,
        'npc_count': metrics['npc_count'],
        'roundtime': npd.ROUNDTIME,
        'seed': args.seed,
        'size': f'{size.lines}x{size.columns}',
        'rounds': completed,
        'tensor_metrics': metrics,
    }


def run(args):
    if args.engine == 'tensor_rank1':
        return run_tensor_rank1(args)

    npd.seed_all(args.seed)
    game = npd.init(npd.Game(
        size=args.size,
        action_backend=args.action_backend,
        action_device=args.action_device,
        batched_min_family_size=args.batched_min_family_size,
    ), num=args.initial_cells)
    countdown = npd.ROUNDTIME
    frames = 0
    completed = []
    device = torch.device(args.action_device) if args.action_device != 'auto' else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    if args.action_backend == npd.ACTION_BACKEND_FAMILY_BATCHED and device.type == 'cuda':
        torch.cuda.synchronize(device)
    round_start = time.perf_counter()
    with torch.inference_mode():
        while len(completed) < args.rounds:
            if len(game.cells) == 0:
                game = npd.init(game, num=npd.MIN_WAVE)
            game, countdown = npd.advance_round(game, countdown)
            game = npd.step(game)
            countdown -= 1
            frames += 1
            if countdown == npd.ROUNDTIME - 1 and frames > 1:
                if args.action_backend == npd.ACTION_BACKEND_FAMILY_BATCHED and device.type == 'cuda':
                    torch.cuda.synchronize(device)
                now = time.perf_counter()
                completed.append({
                    'round': game.rounds,
                    'seconds': now - round_start,
                    'cells': len(game.cells),
                })
                round_start = now
    return {
        'engine': args.engine,
        'action_backend': args.action_backend,
        'action_device': args.action_device,
        'batched_min_family_size': args.batched_min_family_size,
        'cuda_name': torch.cuda.get_device_name(device) if device.type == 'cuda' and torch.cuda.is_available() else '',
        'initial_cells': args.initial_cells,
        'roundtime': npd.ROUNDTIME,
        'seed': args.seed,
        'size': f'{npd.terminal_size(args.size).lines}x{npd.terminal_size(args.size).columns}',
        'rounds': completed,
    }


def main():
    args = parse_args()
    metrics = run(args)
    if not args.quiet:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
            handle.write('\n')


if __name__ == '__main__':
    main()
