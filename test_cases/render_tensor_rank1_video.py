#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import neural_petri_dish as npd
from tensor_rank1_sim import (
    COMPILE_MODES,
    HEALTH_DTYPES,
    MATMUL_PRECISIONS,
    CudaGraphFamilyBasisBlockRunner,
    TensorRank1State,
    resolve_device,
    resolve_health_dtype,
    synchronize,
)


DEFAULT_OUTPUT = 'test_cases/artifacts/tensor_rank1_10k_rounds_every_1000.mp4'


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('value must be positive')
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description='Render sampled rounds from the tensor rank-1 engine.')
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--rounds', type=positive_int, default=10000)
    parser.add_argument('--render-rounds', type=positive_int, default=10)
    parser.add_argument('--round-stride', '--save-every-rounds', dest='round_stride', type=positive_int, default=1000)
    parser.add_argument('--fps', type=positive_int, default=24)
    parser.add_argument('--size', type=npd.parse_size, default=(60, 80), help='grid size as LINESxCOLUMNS')
    parser.add_argument('--initial-cells', type=positive_int, default=2500)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--action-device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--cell-size', type=positive_int, default=8)
    parser.add_argument('--status-height', type=positive_int, default=34)
    parser.add_argument('--tensor-block-steps', type=positive_int, default=10)
    parser.add_argument('--tensor-family-capacity', type=positive_int, default=10)
    parser.add_argument('--tensor-cell-capacity', type=positive_int)
    parser.add_argument('--tensor-initial-health', type=positive_int, default=15)
    parser.add_argument('--tensor-wave-initial-health', type=positive_int, default=2)
    parser.add_argument('--tensor-coeff-scale', type=float, default=npd.FACTORED_WAVE_COEFF_SCALE)
    parser.add_argument('--tensor-stationary-health-cap', type=int, default=0)
    parser.add_argument('--tensor-static-refill-check-every', type=positive_int, default=100)
    parser.add_argument('--tensor-health-dtype', choices=tuple(HEALTH_DTYPES), default='int32')
    parser.add_argument('--tensor-compile-mode', choices=COMPILE_MODES, default='default')
    parser.add_argument('--tensor-matmul-precision', choices=MATMUL_PRECISIONS, default='high')
    parser.add_argument('--tensor-cuda-graph', dest='no_tensor_cuda_graph', action='store_false')
    parser.add_argument('--no-tensor-cuda-graph', dest='no_tensor_cuda_graph', action='store_true')
    parser.add_argument('--save-final-state', help='write final tensor state/debug payload with torch.save')
    parser.set_defaults(no_tensor_cuda_graph=True)
    args = parser.parse_args()
    if args.tensor_static_refill_check_every % args.tensor_block_steps != 0:
        parser.error('--tensor-block-steps must divide --tensor-static-refill-check-every')
    return args


def selected_rounds(render_rounds, round_stride):
    return [round_index * round_stride for round_index in range(render_rounds)]


def tensor_status_text(health, rounds, countdown, global_frame):
    active_health = health[health > 0]
    if active_health.size:
        avghealth = round(float(active_health.mean()))
        maxhealth = int(active_health.max())
    else:
        avghealth = 0
        maxhealth = 0
    return (
        f'Frame {global_frame}  AVG HP {avghealth}  MAX HP {maxhealth}  '
        f'Cells {active_health.size}  Rounds {rounds}  Countdown {countdown}'
    )


def render_tensor_frame(index_grid, health, size, rounds, countdown, global_frame, cell_size, status_height, font):
    visible = index_grid[2:size.lines, 2:size.columns + 2]
    rows, cols = visible.shape
    cells = np.full((rows, cols, 3), (8, 10, 14), dtype=np.uint8)

    active = visible >= 0
    if np.any(active):
        ids = visible[active].astype(np.int64)
        colors = np.empty((ids.shape[0], 3), dtype=np.uint8)
        colors[:, 0] = 48 + ((ids * 53) % 176)
        colors[:, 1] = 80 + ((ids * 97) % 144)
        colors[:, 2] = 64 + ((ids * 29) % 160)
        cells[active] = colors

    frame = np.repeat(np.repeat(cells, cell_size, axis=0), cell_size, axis=1)
    image = Image.new('RGB', (cols * cell_size, rows * cell_size + status_height), (14, 18, 24))
    image.paste(Image.fromarray(frame), (0, 0))
    draw = ImageDraw.Draw(image)
    status_y = rows * cell_size
    draw.rectangle((0, status_y, cols * cell_size, rows * cell_size + status_height), fill=(14, 18, 24))
    draw.text(
        (6, status_y + 9),
        tensor_status_text(health, rounds, countdown, global_frame),
        fill=(232, 238, 246),
        font=font,
    )
    return np.asarray(image)


class TensorRank1VideoRun:
    def __init__(self, args):
        npd.seed_all(args.seed)
        self.args = args
        size = npd.terminal_size(args.size)
        device_name = args.action_device
        if device_name == 'auto':
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = resolve_device(device_name)
        self.size = size
        self.rounds = 0
        self.countdown = npd.ROUNDTIME
        self.frame = 0
        self.waves_spawned = 0
        self.empty_refills = 0
        self.cuda_graph_captures = 0
        self.active_family_count = 1
        self.last_active_cells = None
        self.graph_runners = {}

        if args.tensor_matmul_precision is not None:
            torch.set_float32_matmul_precision(args.tensor_matmul_precision)
        board_capacity = (size.lines - 2) * size.columns
        initial_cells = min(args.initial_cells, board_capacity)
        self.state = TensorRank1State.fixed_capacity(
            active_cells=initial_cells,
            height=size.lines,
            width=size.columns,
            active_families=1,
            family_capacity=args.tensor_family_capacity,
            device=self.device,
            initial_health=args.tensor_initial_health,
            cell_capacity=args.tensor_cell_capacity,
            health_dtype=resolve_health_dtype(args.tensor_health_dtype),
            coeff_scale=args.tensor_coeff_scale,
            stationary_health_cap=args.tensor_stationary_health_cap,
        )
        self.warm_compiled_shape(args.tensor_block_steps)

    def warm_compiled_shape(self, step_count):
        if self.device.type != 'cuda':
            return
        compile_state = self.state.clone()
        compile_state.compiled_snapshot_combat_steps(
            step_count,
            rebuild_grid=True,
            family_basis=True,
            compile_mode=self.args.tensor_compile_mode,
        )
        synchronize(self.device)

    def active_cell_count(self):
        if self.last_active_cells is None:
            self.last_active_cells = int((self.state.health > 0).sum().item())
        return self.last_active_cells

    def invalidate_active_count(self):
        self.last_active_cells = None

    def apply_spawn_count(self, spawned):
        if self.last_active_cells is not None:
            self.last_active_cells += int(spawned)

    def spawn_wave(self, count):
        previous_family_version = self.state.family_capacity_version()
        spawned, self.active_family_count = self.state.append_static_weighted_wave(
            self.active_family_count,
            count,
            initial_health=self.args.tensor_wave_initial_health,
            coeff_scale=self.args.tensor_coeff_scale,
        )
        if self.state.family_capacity_version() != previous_family_version:
            self.graph_runners.clear()
        self.apply_spawn_count(spawned)
        self.waves_spawned += spawned
        return spawned

    def maybe_refill_empty(self):
        if self.frame % self.args.tensor_static_refill_check_every != 0:
            return True
        if self.active_cell_count() != 0:
            return True
        spawned = self.spawn_wave(npd.MIN_WAVE)
        self.empty_refills += 1
        return spawned > 0

    def graph_runner(self, step_count):
        runner = self.graph_runners.get(step_count)
        if runner is None:
            self.warm_compiled_shape(step_count)
            runner = CudaGraphFamilyBasisBlockRunner(
                self.state,
                step_count,
                self.args.tensor_compile_mode,
            )
            self.graph_runners[step_count] = runner
            self.cuda_graph_captures += 1
        return runner

    def run_steps(self, step_count, use_cuda_graph=True):
        if step_count <= 0:
            return
        if self.device.type == 'cuda':
            if use_cuda_graph and not self.args.no_tensor_cuda_graph:
                self.graph_runner(step_count).replay()
            else:
                self.state.compiled_snapshot_combat_steps(
                    step_count,
                    rebuild_grid=True,
                    family_basis=True,
                    compile_mode=self.args.tensor_compile_mode,
                )
        else:
            for _ in range(step_count):
                self.state.step(movement='snapshot_combat', compact_dead=False, sync_positions=False)
        self.invalidate_active_count()

    def finish_round_if_needed(self):
        if self.countdown != 0:
            return
        wave_size = max(npd.PER_WAVE - self.active_cell_count(), npd.MIN_WAVE)
        self.spawn_wave(wave_size)
        self.countdown = npd.ROUNDTIME
        self.rounds += 1

    def advance_unrendered_round(self):
        while self.countdown > 0:
            if not self.maybe_refill_empty():
                return False
            frames_to_refill_check = (
                self.args.tensor_static_refill_check_every
                - (self.frame % self.args.tensor_static_refill_check_every)
            )
            step_count = min(self.args.tensor_block_steps, self.countdown, frames_to_refill_check)
            self.run_steps(step_count, use_cuda_graph=True)
            self.frame += step_count
            self.countdown -= step_count
        self.finish_round_if_needed()
        return True

    def snapshot(self):
        synchronize(self.device)
        return (
            self.state.index_grid.detach().cpu().numpy(),
            self.state.health.detach().cpu().numpy(),
        )

    def render_round(self, writer, font):
        frames_written = 0
        while self.countdown > 0:
            if not self.maybe_refill_empty():
                return frames_written, False
            index_grid, health = self.snapshot()
            writer.append_data(render_tensor_frame(
                index_grid,
                health,
                self.size,
                self.rounds,
                self.countdown,
                self.frame,
                self.args.cell_size,
                self.args.status_height,
                font,
            ))
            frames_written += 1
            self.run_steps(1, use_cuda_graph=False)
            self.frame += 1
            self.countdown -= 1
        self.finish_round_if_needed()
        return frames_written, True

    def metrics(self, elapsed_seconds, frames_written, rendered_rounds, output):
        synchronize(self.device)
        return {
            'active_cells_final': self.active_cell_count(),
            'action_device': str(self.device),
            'cell_capacity': self.state.cells,
            'cuda_graph_captures': self.cuda_graph_captures if self.device.type == 'cuda' else None,
            'cuda_name': torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else '',
            'elapsed_seconds': elapsed_seconds,
            'empty_refills': self.empty_refills,
            'family_capacity_final': self.state.families,
            'frames_written': frames_written,
            'full_simulation_frames': self.frame,
            'output': str(output),
            'rounds_completed': self.rounds,
            'rendered_rounds': rendered_rounds,
            'waves_spawned': self.waves_spawned,
        }

    def save_state(self, path, metrics):
        synchronize(self.device)
        state = {}
        for field_name in self.state.__dataclass_fields__:
            value = getattr(self.state, field_name)
            if isinstance(value, torch.Tensor):
                state[field_name] = value.detach().cpu()
            else:
                state[field_name] = value
        payload = {
            'args': vars(self.args),
            'metrics': metrics,
            'progress': {
                'rounds': self.rounds,
                'countdown': self.countdown,
                'frame': self.frame,
                'active_family_count': self.active_family_count,
                'waves_spawned': self.waves_spawned,
                'empty_refills': self.empty_refills,
            },
            'state': state,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return path


def write_manifest(path, args, metrics):
    manifest = [
        'Neural Petri Dish tensor rank-1 long-run video artifact',
        '',
        f'output: {path}',
        f'elapsed_seconds: {metrics["elapsed_seconds"]:.6f}',
        f'rounds_requested: {args.rounds}',
        f'rounds_completed: {metrics["rounds_completed"]}',
        f'render_rounds: {args.render_rounds}',
        f'round_stride: {args.round_stride}',
        f'rendered_rounds: {",".join(str(round_num) for round_num in metrics["rendered_rounds"])}',
        f'frames_written: {metrics["frames_written"]}',
        f'full_simulation_frames: {metrics["full_simulation_frames"]}',
        f'fps: {args.fps}',
        f'size: {npd.terminal_size(args.size).lines}x{npd.terminal_size(args.size).columns}',
        f'initial_cells: {args.initial_cells}',
        f'seed: {args.seed}',
        f'action_device: {metrics["action_device"]}',
        f'cuda_name: {metrics["cuda_name"]}',
        f'tensor_block_steps: {args.tensor_block_steps}',
        f'tensor_family_capacity: {args.tensor_family_capacity}',
        f'tensor_health_dtype: {args.tensor_health_dtype}',
        f'tensor_coeff_scale: {args.tensor_coeff_scale}',
        f'tensor_stationary_health_cap: {args.tensor_stationary_health_cap}',
        f'tensor_compile_mode: {args.tensor_compile_mode}',
        f'tensor_matmul_precision: {args.tensor_matmul_precision}',
        f'cuda_graph_captures: {metrics["cuda_graph_captures"]}',
        f'family_capacity_final: {metrics["family_capacity_final"]}',
        f'active_cells_final: {metrics["active_cells_final"]}',
        f'waves_spawned: {metrics["waves_spawned"]}',
        f'final_state: {metrics.get("final_state", "")}',
        f'empty_refills: {metrics["empty_refills"]}',
        '',
    ]
    path.with_suffix(path.suffix + '.manifest.txt').write_text('\n'.join(manifest), encoding='utf-8')
    path.with_suffix(path.suffix + '.metrics.json').write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rounds_to_render = selected_rounds(args.render_rounds, args.round_stride)
    rounds_to_render_set = set(rounds_to_render)
    rendered_rounds = []
    frames_written = 0
    font = ImageFont.load_default()

    started = time.perf_counter()
    with torch.no_grad(), imageio.get_writer(output, fps=args.fps, macro_block_size=1) as writer:
        run = TensorRank1VideoRun(args)
        while run.rounds < args.rounds:
            if run.rounds in rounds_to_render_set:
                rendered_rounds.append(run.rounds)
                written, ok = run.render_round(writer, font)
                frames_written += written
            else:
                ok = run.advance_unrendered_round()
            if not ok:
                break

        elapsed = time.perf_counter() - started
        metrics = run.metrics(elapsed, frames_written, rendered_rounds, output)
        if args.save_final_state:
            state_path = run.save_state(args.save_final_state, metrics)
            metrics['final_state'] = str(state_path)

    write_manifest(output, args, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
