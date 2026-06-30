#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('value must be positive')
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run Neural Petri Dish snapshots and optionally ask codex exec to review them.'
    )
    parser.add_argument('--frames', type=positive_int, default=80)
    parser.add_argument('--snapshot-every', type=positive_int, default=10)
    parser.add_argument('--size', default='18x60')
    parser.add_argument('--initial-cells', type=positive_int, default=140)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output-dir', default='test_cases/artifacts/vibe_snapshot')
    parser.add_argument('--no-codex-review', action='store_true')
    parser.add_argument(
        '--reasoning',
        choices=['low', 'medium', 'high', 'xhigh'],
        default='high',
        help='Codex model_reasoning_effort for the snapshot review.',
    )
    parser.add_argument('--model', help='Optional Codex model override.')
    parser.add_argument('--keep-existing', action='store_true')
    return parser.parse_args()


def run_command(command, cwd, stdout_path=None, stderr_path=None):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if stdout_path:
        stdout_path.write_text(result.stdout, encoding='utf-8')
    if stderr_path:
        stderr_path.write_text(result.stderr, encoding='utf-8')
    if result.returncode != 0:
        raise SystemExit(
            f'command failed with exit code {result.returncode}: {" ".join(command)}\n{result.stderr}'
        )
    return result


def simulation_command(repo_root, args, output_dir):
    return [
        sys.executable,
        str(repo_root / 'neural_petri_dish.py'),
        '--max-frames',
        str(args.frames),
        '--snapshot-every',
        str(args.snapshot_every),
        '--snapshot-dir',
        str(output_dir),
        '--no-render',
        '--frame-rate',
        '0',
        '--size',
        args.size,
        '--initial-cells',
        str(args.initial_cells),
        '--seed',
        str(args.seed),
    ]


def expected_snapshot_count(args):
    return ((args.frames - 1) // args.snapshot_every) + 1


def write_run_manifest(output_dir, args, command):
    manifest = [
        'Neural Petri Dish snapshot review run',
        '',
        f'frames: {args.frames}',
        f'snapshot_every: {args.snapshot_every}',
        f'expected_snapshots: {expected_snapshot_count(args)}',
        f'size: {args.size}',
        f'initial_cells: {args.initial_cells}',
        f'seed: {args.seed}',
        '',
        'simulation_command:',
        ' '.join(command),
        '',
    ]
    path = output_dir / 'run_manifest.txt'
    path.write_text('\n'.join(manifest), encoding='utf-8')
    return path


def run_simulation(repo_root, args, output_dir):
    command = simulation_command(repo_root, args, output_dir)
    write_run_manifest(output_dir, args, command)
    run_command(
        command,
        cwd=repo_root,
        stdout_path=output_dir / 'simulation_stdout.log',
        stderr_path=output_dir / 'simulation_stderr.log',
    )
    return command


def review_with_codex(repo_root, args, output_dir, command, snapshots):
    if shutil.which('codex') is None:
        raise SystemExit('codex executable was not found on PATH')

    review_path = output_dir / 'codex_review.md'
    snapshot_list = '\n'.join(f'- {path.name}' for path in snapshots)
    prompt = f'''
You are reviewing a bounded, vibe-style run of the Neural Petri Dish terminal simulation.
Treat this as an integration sanity check, not a unit test.

Source under review:
- {repo_root / 'neural_petri_dish.py'}

Run configuration:
- frames: {args.frames}
- snapshot_every: {args.snapshot_every}
- expected_snapshots: {expected_snapshot_count(args)}
- grid_size: {args.size}
- initial_cells: {args.initial_cells}
- seed: {args.seed}
- command: {' '.join(command)}

Snapshot artifacts:
- {output_dir}
- run_manifest.txt
{snapshot_list}

Assess whether the snapshots look consistent with the implementation. Focus on:
- The number of snapshot files and the frame numbers match snapshot_every.
- Plain-text grid dimensions match the configured grid size and source rendering logic.
- Status lines are plausible for the source, especially Total Players, health fields, rounds, and countdown.
- Cells appear to change over time in a way compatible with movement, collision damage, death, and reproduction.
- There are no obvious impossible states: blank frames after a nonempty initialization, cells outside the rendered play area, duplicate-looking occupancy anomalies, unhandled tracebacks, or contradicting logs.

Return a concise PASS/FAIL review with:
- verdict
- evidence inspected
- any suspicious behavior with source-backed reasoning
- residual limits of this vibe test

Do not edit files or run write commands.
'''.strip()

    command = [
        'codex',
        '--ask-for-approval',
        'never',
        'exec',
        '-C',
        str(repo_root),
        '-c',
        f'model_reasoning_effort="{args.reasoning}"',
        '--sandbox',
        'read-only',
        '--ephemeral',
        '--output-last-message',
        str(review_path),
    ]
    if args.model:
        command.extend(['--model', args.model])
    command.append(prompt)

    run_command(
        command,
        cwd=repo_root,
        stdout_path=output_dir / 'codex_stdout.log',
        stderr_path=output_dir / 'codex_stderr.log',
    )
    return review_path


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()
    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = run_simulation(repo_root, args, output_dir)
    snapshots = sorted(output_dir.glob('frame_*.txt'))
    if not snapshots:
        raise SystemExit('simulation completed but produced no snapshots')
    expected = expected_snapshot_count(args)
    if len(snapshots) != expected:
        raise SystemExit(f'expected {expected} snapshots, found {len(snapshots)}')

    print(f'wrote {len(snapshots)} snapshots to {output_dir}')
    if args.no_codex_review:
        return

    review_path = review_with_codex(repo_root, args, output_dir, command, snapshots)
    print(f'wrote codex review to {review_path}')


if __name__ == '__main__':
    main()
