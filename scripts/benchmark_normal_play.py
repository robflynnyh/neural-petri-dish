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
    HEALTH_DTYPES,
    MATMUL_PRECISIONS,
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
    parser.add_argument('--tensor-stationary-health-cap', type=int, default=1)
    parser.add_argument('--tensor-family-capacity', type=positive_int, default=10)
    parser.add_argument('--tensor-cell-capacity', type=positive_int)
    parser.add_argument('--tensor-static-refill-check-every', type=positive_int, default=100)
    parser.add_argument('--tensor-health-dtype', choices=tuple(HEALTH_DTYPES), default='float32')
    parser.add_argument('--tensor-matmul-precision', choices=MATMUL_PRECISIONS, default='high')
    parser.add_argument('--tensor-compile-mode', choices=COMPILE_MODES, default='default')
    parser.add_argument('--tensor-compiled-block-steps', type=positive_int, default=10)
    parser.add_argument('--tensor-cuda-graph-block', dest='no_tensor_cuda_graph_block', action='store_false')
    parser.add_argument('--no-tensor-cuda-graph-block', dest='no_tensor_cuda_graph_block', action='store_true')
    parser.set_defaults(no_tensor_cuda_graph_block=True)
    parser.add_argument('--output-json')
    args = parser.parse_args()
    if args.engine == 'tensor_rank1' and args.action_device == 'cpu':
        parser.error('--engine tensor_rank1 requires --action-device cuda or --action-device auto')
    return args


def run_tensor_rank1(args):
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
    )
    completed = []
    for index, segment in enumerate(metrics.get('trace_segments', []), start=1):
        completed.append({
            'round': index,
            'seconds': segment['seconds'],
            'cells': segment['active_cells_end'],
            'waves_spawned': segment['waves_spawned'],
            'empty_refills': segment['empty_refills'],
        })
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
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
            handle.write('\n')


if __name__ == '__main__':
    main()
