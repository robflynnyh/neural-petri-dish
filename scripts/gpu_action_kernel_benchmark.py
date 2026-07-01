#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError('value must be non-negative')
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark GPU-resident rank-1 action computation with tensor neighbor extraction.'
    )
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--cells', type=nonnegative_int, default=200000)
    parser.add_argument('--height', type=positive_int, default=512)
    parser.add_argument('--width', type=positive_int, default=512)
    parser.add_argument('--steps', type=positive_int, default=500)
    parser.add_argument(
        '--warmup-steps',
        type=int,
        help='warmup steps before timing; defaults to 0 for --compiled-block-steps > 1 and 20 otherwise',
    )
    parser.add_argument('--families', type=positive_int, default=1)
    parser.add_argument('--initial-health', type=positive_int, default=2)
    parser.add_argument('--health-dtype', choices=tuple(HEALTH_DTYPES), default='int64')
    parser.add_argument(
        '--matmul-precision',
        choices=MATMUL_PRECISIONS,
        help='optional torch float32 matmul precision setting; high enables TF32 matmuls on supported NVIDIA GPUs',
    )
    parser.add_argument('--wave-every', type=int, default=0)
    parser.add_argument('--wave-size', type=int, default=0)
    parser.add_argument('--wave-initial-health', type=positive_int, default=2)
    parser.add_argument(
        '--compact-every',
        type=int,
        default=1,
        help='compact dead cells every N steps; 0 disables scheduled compaction except wave-boundary compaction',
    )
    parser.add_argument(
        '--checksum-actions',
        type=int,
        default=1024,
        help='number of leading actions to reduce into action_checksum each step; use 0 for pure timing',
    )
    parser.add_argument(
        '--trace-every',
        type=int,
        default=0,
        help='emit synchronized timing/state trace segments every N steps; 0 disables tracing',
    )
    parser.add_argument(
        '--compiled-step',
        action='store_true',
        help='use torch.compile for fixed-shape CUDA snapshot_combat probes',
    )
    parser.add_argument(
        '--compile-mode',
        choices=COMPILE_MODES,
        default='reduce-overhead',
        help='torch.compile mode for --compiled-step',
    )
    parser.add_argument(
        '--compiled-block-steps',
        type=positive_int,
        default=1,
        help='run this many fixed-shape compiled steps per Python call; currently requires static rebuild-grid family-basis mode',
    )
    parser.add_argument(
        '--cuda-graph-block',
        action='store_true',
        help='capture each compiled fixed-shape block in a CUDA graph and replay it during timing',
    )
    parser.add_argument(
        '--static-capacity',
        action='store_true',
        help='keep cell/family tensor shapes fixed so compiled snapshot_combat can run across scheduled waves',
    )
    parser.add_argument(
        '--family-capacity',
        type=positive_int,
        help='preallocated family rows for --static-capacity; defaults to enough rows for scheduled waves',
    )
    parser.add_argument(
        '--cell-capacity',
        type=positive_int,
        help='preallocated cell slots for --static-capacity; defaults to every playable grid square',
    )
    parser.add_argument(
        '--static-refill-empty',
        action='store_true',
        help='for --static-capacity, refill inactive slots when the active health mask becomes empty',
    )
    parser.add_argument(
        '--static-refill-check-every',
        type=positive_int,
        default=1,
        help='check active-mask emptiness every N steps when --static-refill-empty is enabled',
    )
    parser.add_argument(
        '--static-rebuild-grid',
        action='store_true',
        help='for compiled static-capacity mode, rebuild playable grid/index arrays each step instead of incremental writes',
    )
    parser.add_argument(
        '--family-basis-step',
        action='store_true',
        help='for compiled static rebuild-grid mode, use family-basis matmuls instead of per-cell bmm',
    )
    parser.add_argument(
        '--movement',
        choices=('none', 'snapshot', 'snapshot_combat'),
        default='none',
        help='include GPU-resident snapshot movement/combat and occupancy-grid rebuild',
    )
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output-json')
    args = parser.parse_args()
    if args.compiled_step:
        if args.device != 'cuda':
            parser.error('--compiled-step requires --device cuda')
        if args.movement != 'snapshot_combat':
            parser.error('--compiled-step requires --movement snapshot_combat')
        if args.compact_every != 0:
            parser.error('--compiled-step requires --compact-every 0')
        if not args.static_capacity and (args.wave_every != 0 or args.wave_size != 0):
            parser.error('--compiled-step requires --wave-every 0 and --wave-size 0 unless --static-capacity is enabled')
    if args.static_capacity:
        if args.movement != 'snapshot_combat':
            parser.error('--static-capacity requires --movement snapshot_combat')
        if args.compact_every != 0:
            parser.error('--static-capacity requires --compact-every 0')
        if args.family_capacity is not None and args.family_capacity < args.families:
            parser.error('--family-capacity must be at least --families')
        if args.cell_capacity is not None and args.cell_capacity < args.cells:
            parser.error('--cell-capacity must be at least --cells')
        if args.static_refill_empty and args.family_capacity is None:
            parser.error('--static-refill-empty requires explicit --family-capacity')
    elif args.static_refill_empty:
        parser.error('--static-refill-empty requires --static-capacity')
    if args.static_rebuild_grid:
        if not args.static_capacity:
            parser.error('--static-rebuild-grid requires --static-capacity')
        if not args.compiled_step:
            parser.error('--static-rebuild-grid requires --compiled-step')
    if args.family_basis_step and not args.static_rebuild_grid:
        parser.error('--family-basis-step requires --static-rebuild-grid')
    if args.compiled_block_steps > 1:
        if not args.compiled_step:
            parser.error('--compiled-block-steps requires --compiled-step')
        if not args.static_capacity:
            parser.error('--compiled-block-steps requires --static-capacity')
        if not (args.static_rebuild_grid and args.family_basis_step):
            parser.error('--compiled-block-steps requires --static-rebuild-grid and --family-basis-step')
        if args.checksum_actions:
            parser.error('--compiled-block-steps requires --checksum-actions 0')
    if args.cuda_graph_block:
        if args.compiled_block_steps <= 1:
            parser.error('--cuda-graph-block requires --compiled-block-steps > 1')
        if not (args.compiled_step and args.static_capacity and args.static_rebuild_grid and args.family_basis_step):
            parser.error('--cuda-graph-block requires compiled static rebuild-grid family-basis mode')
    if args.warmup_steps is None:
        args.warmup_steps = 0 if args.compiled_block_steps > 1 else 20
    return args


def run(args):
    import torch

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    return benchmark_tensor_state(
        cells=args.cells,
        height=args.height,
        width=args.width,
        families=args.families,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        movement=args.movement,
        device=device,
        initial_health=args.initial_health,
        health_dtype=args.health_dtype,
        wave_every=args.wave_every,
        wave_size=args.wave_size,
        wave_initial_health=args.wave_initial_health,
        compact_every=args.compact_every,
        checksum_actions=args.checksum_actions,
        trace_every=args.trace_every,
        compiled_step=args.compiled_step,
        static_capacity=args.static_capacity,
        family_capacity=args.family_capacity,
        cell_capacity=args.cell_capacity,
        static_refill_empty=args.static_refill_empty,
        static_refill_check_every=args.static_refill_check_every,
        static_rebuild_grid=args.static_rebuild_grid,
        family_basis_step=args.family_basis_step,
        matmul_precision=args.matmul_precision,
        compile_mode=args.compile_mode,
        compiled_block_steps=args.compiled_block_steps,
        cuda_graph_block=args.cuda_graph_block,
    )


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
