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
import neural_petri_dish as npd


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
        description='Estimate normal-size simulator round workload with the fast GPU-resident rank-1 tensor engine.'
    )
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--cells', type=nonnegative_int, default=2500)
    parser.add_argument('--height', type=positive_int, default=60)
    parser.add_argument('--width', type=positive_int, default=80)
    parser.add_argument('--rounds', type=positive_int, default=3)
    parser.add_argument('--roundtime', type=positive_int, default=500)
    parser.add_argument('--families', type=positive_int, default=1)
    parser.add_argument('--initial-health', type=positive_int, default=15)
    parser.add_argument('--per-wave', type=positive_int, default=npd.PER_WAVE)
    parser.add_argument('--min-wave', type=positive_int, default=npd.MIN_WAVE)
    parser.add_argument('--wave-initial-health', type=positive_int, default=2)
    parser.add_argument('--family-capacity', type=positive_int, default=7)
    parser.add_argument('--static-refill-check-every', type=positive_int, default=100)
    parser.add_argument('--health-dtype', choices=tuple(HEALTH_DTYPES), default='int32')
    parser.add_argument('--matmul-precision', choices=MATMUL_PRECISIONS, default='high')
    parser.add_argument('--compile-mode', choices=COMPILE_MODES, default='default')
    parser.add_argument('--compiled-block-steps', type=positive_int, default=50)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output-json')
    parser.add_argument(
        '--no-cuda-graph-block',
        action='store_true',
        help='disable CUDA graph replay around the compiled fixed-shape block',
    )
    parser.add_argument(
        '--json-only',
        action='store_true',
        help='print only JSON metrics, without the human-readable round table',
    )
    args = parser.parse_args()
    if args.device != 'cuda':
        parser.error('this normal-round tensor benchmark currently targets --device cuda')
    if args.family_capacity < args.families:
        parser.error('--family-capacity must be at least --families')
    if args.compiled_block_steps > args.roundtime:
        parser.error('--compiled-block-steps must not exceed --roundtime')
    return args


def round_summaries(metrics):
    rounds = []
    for index, segment in enumerate(metrics.get('trace_segments', []), start=1):
        rounds.append({
            'round': index,
            'start_step': segment['start_step'],
            'end_step': segment['end_step'],
            'seconds': segment['seconds'],
            'active_cells_start': segment['active_cells_start'],
            'active_cells_end': segment['active_cells_end'],
            'cells_start': segment['cells_start'],
            'cells_end': segment['cells_end'],
            'families_start': segment['families_start'],
            'families_end': segment['families_end'],
            'waves_spawned': segment['waves_spawned'],
            'waves_spawned_total': segment['waves_spawned_total'],
            'empty_refills': segment['empty_refills'],
            'empty_refills_total': segment['empty_refills_total'],
            'processed_cell_steps': segment['processed_cell_steps'],
            'cells_per_second': segment['cells_per_second'],
        })
    return rounds


def build_report(args, metrics):
    rounds = round_summaries(metrics)
    return {
        'benchmark': 'tensor_projected_normal_rounds',
        'engine': 'static_capacity_family_basis_compiled_cuda_graph'
        if metrics.get('cuda_graph_block') else 'static_capacity_family_basis_compiled',
        'semantic_note': (
            'GPU-resident tensor projection for normal-size rounds; current interactive normal play still uses '
            'the Python Game/Cell path.'
        ),
        'device': metrics['device'],
        'cuda_name': metrics['cuda_name'],
        'seed': args.seed,
        'roundtime': args.roundtime,
        'per_wave': args.per_wave,
        'min_wave': args.min_wave,
        'rounds_requested': args.rounds,
        'elapsed_seconds': metrics['elapsed_seconds'],
        'active_cells_final': metrics['active_cells_final'],
        'active_families_final': metrics['active_families_final'],
        'waves_spawned': metrics['waves_spawned'],
        'empty_refills': metrics['empty_refills'],
        'cell_capacity': metrics['cell_capacity'],
        'family_capacity': metrics['family_capacity'],
        'rounds': rounds,
        'raw_metrics': metrics,
    }


def print_table(report):
    print(
        f"benchmark={report['benchmark']} device={report['device']} cuda={report['cuda_name']} "
        f"roundtime={report['roundtime']} total={report['elapsed_seconds']:.6f}s"
    )
    print('round  steps       seconds    active_start  active_end  waves  refills')
    for item in report['rounds']:
        step_range = f"{item['start_step']}-{item['end_step']}"
        print(
            f"{item['round']:>5}  {step_range:<10}  {item['seconds']:>8.6f}  "
            f"{item['active_cells_start']:>12}  {item['active_cells_end']:>10}  "
            f"{item['waves_spawned']:>5}  {item['empty_refills']:>7}"
        )


def run(args):
    import torch

    torch.manual_seed(args.seed)
    metrics = benchmark_tensor_state(
        cells=args.cells,
        height=args.height,
        width=args.width,
        families=args.families,
        steps=args.rounds * args.roundtime,
        warmup_steps=0,
        movement='snapshot_combat',
        device=resolve_device(args.device),
        initial_health=args.initial_health,
        health_dtype=args.health_dtype,
        wave_every=args.roundtime,
        wave_size=args.per_wave,
        wave_initial_health=args.wave_initial_health,
        compact_every=0,
        checksum_actions=0,
        trace_every=args.roundtime,
        compiled_step=True,
        static_capacity=True,
        family_capacity=args.family_capacity,
        static_refill_empty=True,
        static_refill_check_every=args.static_refill_check_every,
        static_rebuild_grid=True,
        family_basis_step=True,
        matmul_precision=args.matmul_precision,
        compile_mode=args.compile_mode,
        compiled_block_steps=args.compiled_block_steps,
        cuda_graph_block=not args.no_cuda_graph_block,
        normal_round_refill=True,
        per_wave=args.per_wave,
        min_wave=args.min_wave,
    )
    return build_report(args, metrics)


def main():
    args = parse_args()
    report = run(args)
    if not args.json_only:
        print_table(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write('\n')


if __name__ == '__main__':
    main()
