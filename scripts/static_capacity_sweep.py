#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensor_rank1_sim import COMPILE_MODES, HEALTH_DTYPES, MATMUL_PRECISIONS, benchmark_tensor_state, resolve_device


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


def int_list(value):
    return [positive_int(item.strip()) for item in value.split(',') if item.strip()]


def dtype_list(value):
    parsed = [item.strip() for item in value.split(',') if item.strip()]
    unknown = [item for item in parsed if item not in HEALTH_DTYPES]
    if unknown:
        raise argparse.ArgumentTypeError(f'unknown health dtype: {unknown[0]}')
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description='Sweep fixed-capacity compiled rank-1 GPU benchmark settings.')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--cells', type=nonnegative_int, default=2500)
    parser.add_argument('--height', type=positive_int, default=60)
    parser.add_argument('--width', type=positive_int, default=80)
    parser.add_argument('--steps', type=positive_int, default=1500)
    parser.add_argument(
        '--warmup-steps',
        type=int,
        help='warmup steps before timing; defaults to 0 for --compiled-block-steps > 1 and 20 otherwise',
    )
    parser.add_argument('--families', type=positive_int, default=1)
    parser.add_argument('--initial-health', type=positive_int, default=15)
    parser.add_argument('--wave-every', type=int, default=500)
    parser.add_argument('--wave-size', type=int, default=300)
    parser.add_argument('--wave-initial-health', type=positive_int, default=2)
    parser.add_argument('--health-dtypes', type=dtype_list, default=['int32'])
    parser.add_argument(
        '--matmul-precision',
        choices=MATMUL_PRECISIONS,
        help='optional torch float32 matmul precision setting; high enables TF32 matmuls on supported NVIDIA GPUs',
    )
    parser.add_argument('--family-capacities', type=int_list, default=[16])
    parser.add_argument(
        '--cell-capacities',
        type=int_list,
        help='comma-separated cell capacities; omit to use full board capacity',
    )
    parser.add_argument('--refill-check-everys', type=int_list, default=[100])
    parser.add_argument('--repeats', type=positive_int, default=1)
    parser.add_argument('--no-compiled-step', action='store_true')
    parser.add_argument(
        '--compile-mode',
        choices=COMPILE_MODES,
        default='reduce-overhead',
        help='torch.compile mode when compiled step is enabled',
    )
    parser.add_argument(
        '--compiled-block-steps',
        type=positive_int,
        default=1,
        help='run this many fixed-shape compiled steps per Python call; requires static rebuild-grid family-basis mode',
    )
    parser.add_argument(
        '--cuda-graph-block',
        action='store_true',
        help='capture each compiled fixed-shape block in a CUDA graph and replay it during timing',
    )
    parser.add_argument('--no-static-refill-empty', action='store_true')
    parser.add_argument('--static-rebuild-grid', action='store_true')
    parser.add_argument('--family-basis-step', action='store_true')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output-json')
    parser.add_argument('--output-csv')
    args = parser.parse_args()
    if not args.no_compiled_step and args.device != 'cuda':
        parser.error('compiled sweep requires --device cuda; use --no-compiled-step for CPU smoke tests')
    if args.static_rebuild_grid and args.no_compiled_step:
        parser.error('--static-rebuild-grid requires compiled step')
    if args.family_basis_step and not args.static_rebuild_grid:
        parser.error('--family-basis-step requires --static-rebuild-grid')
    if args.compiled_block_steps > 1:
        if args.no_compiled_step:
            parser.error('--compiled-block-steps requires compiled step')
        if not (args.static_rebuild_grid and args.family_basis_step):
            parser.error('--compiled-block-steps requires --static-rebuild-grid and --family-basis-step')
    if args.cuda_graph_block:
        if args.compiled_block_steps <= 1:
            parser.error('--cuda-graph-block requires --compiled-block-steps > 1')
        if args.no_compiled_step or not (args.static_rebuild_grid and args.family_basis_step):
            parser.error('--cuda-graph-block requires compiled static rebuild-grid family-basis mode')
    if args.warmup_steps is None:
        args.warmup_steps = 0 if args.compiled_block_steps > 1 else 20
    return args


def run_case(args, device, health_dtype, family_capacity, cell_capacity, refill_check_every, repeat):
    import torch

    torch.manual_seed(args.seed + repeat)
    return benchmark_tensor_state(
        cells=args.cells,
        height=args.height,
        width=args.width,
        families=args.families,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        movement='snapshot_combat',
        device=device,
        initial_health=args.initial_health,
        health_dtype=health_dtype,
        wave_every=args.wave_every,
        wave_size=args.wave_size,
        wave_initial_health=args.wave_initial_health,
        compact_every=0,
        checksum_actions=0,
        compiled_step=not args.no_compiled_step,
        static_capacity=True,
        family_capacity=family_capacity,
        cell_capacity=cell_capacity,
        static_refill_empty=not args.no_static_refill_empty,
        static_refill_check_every=refill_check_every,
        static_rebuild_grid=args.static_rebuild_grid,
        family_basis_step=args.family_basis_step,
        matmul_precision=args.matmul_precision,
        compile_mode=args.compile_mode,
        compiled_block_steps=args.compiled_block_steps,
        cuda_graph_block=args.cuda_graph_block,
    )


def main():
    args = parse_args()
    device = resolve_device(args.device)
    cell_capacities = args.cell_capacities or [None]
    rows = []
    for health_dtype in args.health_dtypes:
        for family_capacity in args.family_capacities:
            for cell_capacity in cell_capacities:
                for refill_check_every in args.refill_check_everys:
                    for repeat in range(args.repeats):
                        metrics = run_case(
                            args,
                            device,
                            health_dtype,
                            family_capacity,
                            cell_capacity,
                            refill_check_every,
                            repeat,
                        )
                        metrics['repeat'] = repeat
                        rows.append(metrics)
                        print(json.dumps(metrics, sort_keys=True))

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rows, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    if args.output_csv:
        output = Path(args.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with output.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
