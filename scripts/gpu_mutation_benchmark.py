#!/usr/bin/env python3
import argparse
import csv
import json
import time
from pathlib import Path

import torch


NEIGHBOR_DIM = 24
HIDDEN_DIM = 64
INPUT_DIM = NEIGHBOR_DIM + HIDDEN_DIM
OUTPUT_DIM = 9
MODES = ('shared_rank1_factored',)
DEFAULT_MODE = 'shared_rank1_factored'


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('value must be positive')
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark batched Neural Petri Dish genome simulation on CPU or GPU.'
    )
    parser.add_argument('--mode', choices=MODES, default=DEFAULT_MODE)
    parser.add_argument('--population', type=positive_int, default=200000)
    parser.add_argument('--steps', type=positive_int, default=200)
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--mutation-scale', type=float, default=1e-4)
    parser.add_argument('--mutation-every', type=positive_int, default=1)
    parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda'))
    parser.add_argument('--dtype', default='float32', choices=('float32', 'float16', 'bfloat16'))
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output-json')
    parser.add_argument('--output-csv')
    return parser.parse_args()


def resolve_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is false')
    return torch.device(name)


def resolve_dtype(name):
    return {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }[name]


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


class FactoredSharedRank1Population:
    """Shared-base genomes with per-cell rank-1 coefficients.

    This fast path is only valid when every genome shares the same base matrix
    and rank-1 directions. Cells differ by scalar coefficients:

        W_i = W_base + coeff_i * outer(u, v)

    Arbitrary unrelated per-cell matrices cannot be represented by this kernel.
    """

    representation = 'shared_base_rank1_coefficients'

    def __init__(self, population, device, dtype, generator):
        self.population = population
        self.device = device
        self.dtype = dtype
        self.base_weight_1 = torch.randn(HIDDEN_DIM, INPUT_DIM, device=device, dtype=dtype, generator=generator)
        self.base_bias_1 = torch.randn(HIDDEN_DIM, device=device, dtype=dtype, generator=generator)
        self.base_weight_2 = torch.randn(OUTPUT_DIM, HIDDEN_DIM, device=device, dtype=dtype, generator=generator)
        self.base_bias_2 = torch.randn(OUTPUT_DIM, device=device, dtype=dtype, generator=generator)
        self.u_1 = torch.randn(HIDDEN_DIM, device=device, dtype=dtype, generator=generator)
        self.v_1 = torch.randn(INPUT_DIM, device=device, dtype=dtype, generator=generator)
        self.u_2 = torch.randn(OUTPUT_DIM, device=device, dtype=dtype, generator=generator)
        self.v_2 = torch.randn(HIDDEN_DIM, device=device, dtype=dtype, generator=generator)
        self.coeff_1 = torch.zeros(population, device=device, dtype=dtype)
        self.coeff_2 = torch.zeros(population, device=device, dtype=dtype)
        self.state = torch.zeros(population, HIDDEN_DIM, device=device, dtype=dtype)
        self.neighbors = torch.empty(population, NEIGHBOR_DIM, device=device, dtype=dtype)
        self._normalize_rank1_directions()

    def _normalize_rank1_directions(self):
        eps = torch.finfo(self.dtype).eps
        rms_1 = torch.outer(self.u_1, self.v_1).square().mean().sqrt().clamp_min(eps)
        rms_2 = torch.outer(self.u_2, self.v_2).square().mean().sqrt().clamp_min(eps)
        self.u_1.div_(rms_1.sqrt())
        self.v_1.div_(rms_1.sqrt())
        self.u_2.div_(rms_2.sqrt())
        self.v_2.div_(rms_2.sqrt())

    def forward_actions(self):
        self.neighbors.bernoulli_(0.25).mul_(2).sub_(1)
        inputs = torch.cat((self.neighbors, self.state), dim=1)
        base_hidden = inputs.matmul(self.base_weight_1.t()).add_(self.base_bias_1)
        rank1_hidden = (inputs.matmul(self.v_1) * self.coeff_1).unsqueeze(1) * self.u_1.unsqueeze(0)
        hidden = base_hidden.add_(rank1_hidden).relu_()
        base_logits = hidden.matmul(self.base_weight_2.t()).add_(self.base_bias_2)
        rank1_logits = (hidden.matmul(self.v_2) * self.coeff_2).unsqueeze(1) * self.u_2.unsqueeze(0)
        logits = base_logits.add_(rank1_logits)
        self.state = hidden.detach()
        return logits.argmax(dim=1)

    def mutate_(self, mode, scale):
        if mode != 'shared_rank1_factored':
            raise ValueError(f'{type(self).__name__} only supports shared_rank1_factored')
        self.coeff_1.add_(torch.randn_like(self.coeff_1), alpha=scale)
        self.coeff_2.add_(torch.randn_like(self.coeff_2), alpha=scale)


def run_benchmark(args):
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    population = FactoredSharedRank1Population(args.population, device, dtype, generator)

    for step in range(max(args.warmup_steps, 0)):
        population.forward_actions()
        if step % args.mutation_every == 0:
            population.mutate_(args.mode, args.mutation_scale)

    synchronize(device)
    started = time.perf_counter()
    action_checksum = 0
    mutation_steps = 0
    for step in range(args.steps):
        actions = population.forward_actions()
        action_checksum += int(actions[:1024].sum().item())
        if step % args.mutation_every == 0:
            population.mutate_(args.mode, args.mutation_scale)
            mutation_steps += 1
    synchronize(device)
    elapsed = time.perf_counter() - started

    simulated_cells = args.population * args.steps
    mutated_genomes = args.population * mutation_steps
    return {
        'mode': args.mode,
        'device': str(device),
        'cuda_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else '',
        'dtype': str(dtype).replace('torch.', ''),
        'representation': population.representation,
        'population': args.population,
        'steps': args.steps,
        'mutation_every': args.mutation_every,
        'mutation_steps': mutation_steps,
        'elapsed_seconds': elapsed,
        'cells_per_second': simulated_cells / elapsed,
        'mutated_genomes_per_second': mutated_genomes / elapsed,
        'action_checksum': action_checksum,
    }


def write_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    with torch.inference_mode():
        metrics = run_benchmark(args)
    print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if args.output_csv:
        write_csv(args.output_csv, metrics)


if __name__ == '__main__':
    main()
