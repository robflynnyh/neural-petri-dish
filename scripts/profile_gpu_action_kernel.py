#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.profiler import ProfilerActivity, profile

from tensor_rank1_sim import (
    COMPILE_MODES,
    HEALTH_DTYPES,
    MATMUL_PRECISIONS,
    TensorRank1State,
    resolve_device,
    resolve_health_dtype,
    synchronize,
)


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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Profile a warmed compiled fixed-capacity rank-1 action/combat block.'
    )
    parser.add_argument('--device', choices=('cuda',), default='cuda')
    parser.add_argument('--cells', type=nonnegative_int, default=2500)
    parser.add_argument('--height', type=positive_int, default=60)
    parser.add_argument('--width', type=positive_int, default=80)
    parser.add_argument('--families', type=positive_int, default=1)
    parser.add_argument('--family-capacity', type=positive_int, default=7)
    parser.add_argument('--cell-capacity', type=positive_int)
    parser.add_argument('--initial-health', type=positive_int, default=15)
    parser.add_argument('--health-dtype', choices=tuple(HEALTH_DTYPES), default='int32')
    parser.add_argument('--matmul-precision', choices=MATMUL_PRECISIONS, default='high')
    parser.add_argument('--compile-mode', choices=COMPILE_MODES, default='default')
    parser.add_argument('--compiled-block-steps', type=positive_int, default=50)
    parser.add_argument('--profile-blocks', type=positive_int, default=1)
    parser.add_argument('--row-limit', type=positive_int, default=40)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output-json')
    return parser.parse_args()


def event_to_row(event):
    return {
        'key': event.key,
        'count': event.count,
        'cpu_time_total_us': event.cpu_time_total,
        'cuda_time_total_us': event.cuda_time_total,
        'self_cpu_time_total_us': event.self_cpu_time_total,
        'self_cuda_time_total_us': event.self_cuda_time_total,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    device = resolve_device(args.device)
    health_dtype = resolve_health_dtype(args.health_dtype)

    state = TensorRank1State.fixed_capacity(
        active_cells=args.cells,
        height=args.height,
        width=args.width,
        active_families=args.families,
        family_capacity=args.family_capacity,
        device=device,
        initial_health=args.initial_health,
        cell_capacity=args.cell_capacity,
        health_dtype=health_dtype,
    )

    warm_state = state.clone()
    warm_state.compiled_snapshot_combat_steps(
        args.compiled_block_steps,
        rebuild_grid=True,
        family_basis=True,
        compile_mode=args.compile_mode,
    )
    synchronize(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        for _ in range(args.profile_blocks):
            state.compiled_snapshot_combat_steps(
                args.compiled_block_steps,
                rebuild_grid=True,
                family_basis=True,
                compile_mode=args.compile_mode,
            )
        synchronize(device)

    events = prof.key_averages()
    table = events.table(sort_by='cuda_time_total', row_limit=args.row_limit)
    print(table)

    if args.output_json:
        rows = [event_to_row(event) for event in events]
        rows.sort(key=lambda row: row['cuda_time_total_us'], reverse=True)
        output = {
            'cells': args.cells,
            'cell_capacity': state.cells,
            'height': args.height,
            'width': args.width,
            'families': args.families,
            'family_capacity': args.family_capacity,
            'health_dtype': args.health_dtype,
            'matmul_precision': args.matmul_precision,
            'compile_mode': args.compile_mode,
            'compiled_block_steps': args.compiled_block_steps,
            'profile_blocks': args.profile_blocks,
            'cuda_name': torch.cuda.get_device_name(device),
            'events': rows,
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
