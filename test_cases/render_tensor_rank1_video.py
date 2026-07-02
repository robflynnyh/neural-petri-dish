#!/usr/bin/env python3
import argparse
import csv
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
    EVENT_COUNT_NAMES,
    HEALTH_DTYPES,
    MATMUL_PRECISIONS,
    NETWORK_DTYPES,
    CudaGraphFamilyBasisBlockRunner,
    TensorRank1State,
    refresh_runtime_constants,
    resolve_device,
    resolve_health_dtype,
    resolve_network_dtype,
    synchronize,
)


DEFAULT_OUTPUT = 'test_cases/artifacts/tensor_rank1_10k_rounds_every_1000.mp4'
DODGE_SUMMARY_WINDOW_ROUNDS = 100


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('value must be positive')
    return parsed


def non_negative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError('value must be non-negative')
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description='Render sampled rounds from the tensor rank-1 engine.')
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--rounds', type=positive_int, default=10000)
    parser.add_argument('--render-rounds', type=non_negative_int, default=10)
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
    parser.add_argument('--fitness-update-lr', type=float, default=npd.FITNESS_UPDATE_LR)
    parser.add_argument('--movement-health-cost', type=float, default=npd.MOVEMENT_HEALTH_COST)
    parser.add_argument('--stationary-health-cost', type=float, default=npd.STATIONARY_HEALTH_COST)
    parser.add_argument('--stationary-damage-after-steps', type=positive_int, default=npd.STATIONARY_DAMAGE_AFTER_STEPS)
    parser.add_argument('--food-health-reward', type=float, default=npd.FOOD_HEALTH_REWARD)
    parser.add_argument('--kill-health-reward', type=float, default=npd.KILL_HEALTH_REWARD)
    parser.add_argument('--round-transition-health-cost', type=float, default=npd.ROUND_TRANSITION_HEALTH_COST)
    parser.add_argument('--per-wave', type=positive_int, default=npd.PER_WAVE)
    parser.add_argument('--min-wave', type=positive_int, default=npd.MIN_WAVE)
    parser.add_argument('--roundtime', type=positive_int, default=npd.ROUNDTIME)
    parser.add_argument('--npc-count', type=int, default=npd.NPC_COUNT)
    parser.add_argument('--tensor-stationary-health-cap', type=int, default=1)
    parser.add_argument('--tensor-static-refill-check-every', type=positive_int, default=100)
    parser.add_argument('--tensor-health-dtype', choices=tuple(HEALTH_DTYPES), default='float32')
    parser.add_argument('--tensor-network-dtype', choices=('auto', *tuple(NETWORK_DTYPES)), default='auto')
    parser.add_argument('--tensor-compile-mode', choices=COMPILE_MODES, default='default')
    parser.add_argument('--tensor-matmul-precision', choices=MATMUL_PRECISIONS, default='high')
    parser.add_argument('--tensor-cuda-graph', dest='no_tensor_cuda_graph', action='store_false')
    parser.add_argument('--no-tensor-cuda-graph', dest='no_tensor_cuda_graph', action='store_true')
    parser.add_argument('--metrics-only', action='store_true', help='run the tensor simulation and write metrics without opening a video writer')
    parser.add_argument('--save-final-state', help='write final tensor state/debug payload with torch.save')
    parser.set_defaults(no_tensor_cuda_graph=False)
    args = parser.parse_args()
    if args.render_rounds == 0 and not args.metrics_only:
        parser.error('--render-rounds 0 requires --metrics-only')
    if args.tensor_static_refill_check_every % args.tensor_block_steps != 0:
        parser.error('--tensor-block-steps must divide --tensor-static-refill-check-every')
    return args


def selected_rounds(render_rounds, round_stride):
    return [round_index * round_stride for round_index in range(render_rounds)]


def dodge_window_summary(round_summaries, start, end):
    rows = round_summaries[start:end]
    if not rows:
        return {}
    active_steps = max(1, sum(row['active_cell_steps'] for row in rows))
    visible_steps = max(1, sum(row['npc_visible_cell_steps'] for row in rows))
    adjacent_steps = max(1, sum(row['npc_adjacent_cell_steps'] for row in rows))
    participant_cells = max(1, sum(row['participant_cells'] for row in rows))
    return {
        'round_start': int(rows[0]['round']),
        'round_end': int(rows[-1]['round']),
        'rounds': int(len(rows)),
        'survival_mean': float(sum(row['survival_mean'] * row['participant_cells'] for row in rows) / participant_cells),
        'frames_elapsed_mean': float(sum(row['frames_elapsed'] for row in rows) / len(rows)),
        'food_eaten_per_round': float(sum(row['food_eaten'] for row in rows) / len(rows)),
        'npc_visible_cell_steps': int(sum(row['npc_visible_cell_steps'] for row in rows)),
        'npc_visible_move_away_rate': float(sum(row['npc_visible_move_away'] for row in rows) / visible_steps),
        'npc_visible_move_toward_rate': float(sum(row['npc_visible_move_toward'] for row in rows) / visible_steps),
        'npc_visible_death_rate': float(sum(row['npc_visible_deaths'] for row in rows) / visible_steps),
        'npc_adjacent_fraction': float(sum(row['npc_adjacent_cell_steps'] for row in rows) / active_steps),
        'npc_adjacent_death_rate': float(sum(row['npc_adjacent_deaths'] for row in rows) / adjacent_steps),
        'npc_adjacent_final_clear_rate': float(sum(row['npc_adjacent_final_clear'] for row in rows) / adjacent_steps),
        'npc_adjacent_move_away_rate': float(sum(row['npc_adjacent_move_away'] for row in rows) / adjacent_steps),
        'npc_adjacent_stayed_put_rate': float(sum(row['npc_adjacent_stayed_put'] for row in rows) / adjacent_steps),
        'npc_kill_rate': float(sum(row['npc_kills'] for row in rows) / active_steps),
        'death_rate': float(sum(row['deaths'] for row in rows) / active_steps),
        'move_success_rate': float(sum(row['move_successes'] for row in rows) / active_steps),
        'stayed_put_rate': float(sum(row['stayed_put'] for row in rows) / active_steps),
    }


def dodge_metric_summary(round_summaries):
    if not round_summaries:
        return {}
    window = min(DODGE_SUMMARY_WINDOW_ROUNDS, len(round_summaries))
    return {
        'window_rounds': int(window),
        'first_window': dodge_window_summary(round_summaries, 0, window),
        'last_window': dodge_window_summary(round_summaries, len(round_summaries) - window, len(round_summaries)),
    }


def tensor_status_text(health, family_index, food_grid, npc_grid, rounds, countdown, global_frame):
    active_health = health[health > 0]
    if active_health.size:
        avghealth = round(float(active_health.mean()))
        maxhealth = int(active_health.max())
        unique_base_genomes = int(np.unique(family_index[health > 0]).size)
    else:
        avghealth = 0
        maxhealth = 0
        unique_base_genomes = 0
    return (
        f'Frame {global_frame}  AVG HP {avghealth}  MAX HP {maxhealth}  '
        f'Cells {active_health.size}  unique base genomes: {unique_base_genomes}  '
        f'Food {int(food_grid.sum())}  NPCs {int(npc_grid.sum())}  Rounds {rounds}  Countdown {countdown}'
    )


def render_tensor_frame(index_grid, health, family_index, food_grid, npc_grid, size, rounds, countdown, global_frame, cell_size, status_height, font):
    visible = index_grid[2:size.lines, 2:size.columns + 2]
    visible_food = food_grid[2:size.lines, 2:size.columns + 2]
    visible_npcs = npc_grid[2:size.lines, 2:size.columns + 2]
    rows, cols = visible.shape
    cells = np.full((rows, cols, 3), (8, 10, 14), dtype=np.uint8)

    food = visible_food > 0
    if np.any(food):
        cells[food] = (224, 190, 72)

    npcs = visible_npcs > 0
    if np.any(npcs):
        cells[npcs] = (245, 64, 82)

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
        tensor_status_text(health, family_index, food_grid, npc_grid, rounds, countdown, global_frame),
        fill=(232, 238, 246),
        font=font,
    )
    return np.asarray(image)


class TensorRank1VideoRun:
    def __init__(self, args):
        npd.seed_all(args.seed)
        npd.FITNESS_UPDATE_LR = args.fitness_update_lr
        npd.MOVEMENT_HEALTH_COST = args.movement_health_cost
        npd.STATIONARY_HEALTH_COST = args.stationary_health_cost
        npd.STATIONARY_DAMAGE_AFTER_STEPS = args.stationary_damage_after_steps
        npd.FOOD_HEALTH_REWARD = args.food_health_reward
        npd.KILL_HEALTH_REWARD = args.kill_health_reward
        npd.ROUND_TRANSITION_HEALTH_COST = args.round_transition_health_cost
        npd.PER_WAVE = args.per_wave
        npd.MIN_WAVE = args.min_wave
        npd.ROUNDTIME = args.roundtime
        npd.NPC_COUNT = args.npc_count
        refresh_runtime_constants()
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
        self.early_ended_rounds = 0
        self.cuda_graph_captures = 0
        self.active_family_count = 1
        self.last_active_cells = None
        self.graph_runners = {}
        self.round_summaries = []

        if args.tensor_matmul_precision is not None:
            torch.set_float32_matmul_precision(args.tensor_matmul_precision)
        board_capacity = (size.lines - 2) * size.columns
        initial_cells = min(args.initial_cells, board_capacity)
        self.round_start_frame = 0
        self.round_start_active_cells = initial_cells
        self.round_ended_early = False
        self.round_early_countdown_remaining = 0
        self.network_dtype = resolve_network_dtype(args.tensor_network_dtype, self.device)
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
            network_dtype=self.network_dtype,
            coeff_scale=args.tensor_coeff_scale,
            stationary_health_cap=args.tensor_stationary_health_cap,
            npc_count=args.npc_count,
        )
        self.behavior_totals_tensor = torch.zeros(len(EVENT_COUNT_NAMES), device=self.device, dtype=torch.float64)
        self.round_event_counts_tensor = torch.zeros(len(EVENT_COUNT_NAMES), device=self.device, dtype=torch.float64)
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

    def record_event_counts(self, event_counts):
        if event_counts is None:
            return
        counts = event_counts.to(device=self.device, dtype=self.behavior_totals_tensor.dtype)
        self.behavior_totals_tensor.add_(counts)
        self.round_event_counts_tensor.add_(counts)

    def record_round_summary(self):
        active_end = self.active_cell_count()
        participants = self.state.round_participants.detach().cpu().numpy().astype(bool)
        survival = self.state.round_survival_steps.detach().cpu().numpy()
        participant_survival = survival[participants]
        if participant_survival.size:
            survival_mean = float(participant_survival.mean())
            survival_max = float(participant_survival.max())
            survival_std = float(participant_survival.std())
            survival_full_count = int((participant_survival >= npd.ROUNDTIME).sum())
        else:
            survival_mean = 0.0
            survival_max = 0.0
            survival_std = 0.0
            survival_full_count = 0
        summary = {
            'round': int(self.rounds),
            'start_frame': int(self.round_start_frame),
            'end_frame': int(self.frame),
            'frames_elapsed': int(self.frame - self.round_start_frame),
            'countdown_remaining': int(self.countdown),
            'ended_early': int(self.round_ended_early),
            'early_countdown_remaining': int(self.round_early_countdown_remaining),
            'started_cells': int(self.round_start_active_cells),
            'ended_cells': int(active_end),
            'participant_cells': int(participant_survival.size),
            'survival_mean': survival_mean,
            'survival_max': survival_max,
            'survival_std': survival_std,
            'survival_full_count': survival_full_count,
            'food_remaining': int(self.state.food_grid.sum().item()),
        }
        event_counts = self.round_event_counts_tensor.detach().cpu().to(torch.long).tolist()
        summary.update({name: int(value) for name, value in zip(EVENT_COUNT_NAMES, event_counts)})
        active_steps = max(1, summary['active_cell_steps'])
        adjacent_steps = max(1, summary['npc_adjacent_cell_steps'])
        participants_count = max(1, summary['participant_cells'])
        summary.update({
            'survivor_fraction': float(summary['ended_cells'] / participants_count),
            'full_survivor_fraction': float(summary['survival_full_count'] / participants_count),
            'food_eaten_per_cell': float(summary['food_eaten'] / participants_count),
            'food_eaten_per_100k_active_steps': float(100000.0 * summary['food_eaten'] / active_steps),
            'move_success_rate': float(summary['move_successes'] / active_steps),
            'attack_hit_rate': float(summary['attack_hits'] / active_steps),
            'attack_kill_rate': float(summary['attack_kills'] / active_steps),
            'border_hit_rate': float(summary['border_hits'] / active_steps),
            'death_rate': float(summary['deaths'] / active_steps),
            'npc_kill_rate': float(summary['npc_kills'] / active_steps),
            'npc_visible_move_away_rate': float(summary['npc_visible_move_away'] / max(1, summary['npc_visible_cell_steps'])),
            'npc_visible_move_toward_rate': float(summary['npc_visible_move_toward'] / max(1, summary['npc_visible_cell_steps'])),
            'npc_visible_death_rate': float(summary['npc_visible_deaths'] / max(1, summary['npc_visible_cell_steps'])),
            'npc_adjacent_fraction': float(summary['npc_adjacent_cell_steps'] / active_steps),
            'npc_adjacent_death_rate': float(summary['npc_adjacent_deaths'] / adjacent_steps),
            'npc_adjacent_final_clear_rate': float(summary['npc_adjacent_final_clear'] / adjacent_steps),
            'npc_adjacent_move_away_rate': float(summary['npc_adjacent_move_away'] / adjacent_steps),
            'npc_adjacent_stayed_put_rate': float(summary['npc_adjacent_stayed_put'] / adjacent_steps),
            'stayed_put_rate': float(summary['stayed_put'] / active_steps),
        })
        self.round_summaries.append(summary)
        self.round_event_counts_tensor.zero_()

    def spawn_wave(self, count, precomputed_family=None):
        previous_family_version = self.state.family_capacity_version()
        spawned, self.active_family_count = self.state.append_static_weighted_wave(
            self.active_family_count,
            count,
            initial_health=self.args.tensor_wave_initial_health,
            coeff_scale=self.args.tensor_coeff_scale,
            precomputed_family=precomputed_family,
        )
        if self.state.family_capacity_version() != previous_family_version:
            self.graph_runners.clear()
        self.apply_spawn_count(spawned)
        self.waves_spawned += spawned
        return spawned

    def round_should_end_early(self):
        return self.active_cell_count() == 0

    def end_round_early(self):
        self.round_ended_early = True
        self.round_early_countdown_remaining = self.countdown
        self.countdown = 0
        self.early_ended_rounds += 1

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
            npc_graph_safe = self.state.npc_flat_positions.numel() == 0
            if use_cuda_graph and not self.args.no_tensor_cuda_graph and npc_graph_safe:
                event_counts = self.graph_runner(step_count).replay()
            else:
                event_counts = self.state.compiled_snapshot_combat_steps(
                    step_count,
                    rebuild_grid=True,
                    family_basis=True,
                    compile_mode=self.args.tensor_compile_mode,
                )
            self.record_event_counts(event_counts)
        else:
            for _ in range(step_count):
                self.state.record_survival_steps(1)
                self.state.step(movement='snapshot_combat', compact_dead=False, sync_positions=False)
        self.invalidate_active_count()

    def finish_round_if_needed(self):
        if self.countdown != 0:
            return
        self.record_round_summary()
        next_family = self.state.weighted_survivor_family()
        self.state.apply_round_transition_health_cost()
        self.invalidate_active_count()
        wave_size = max(npd.PER_WAVE - self.active_cell_count(), npd.MIN_WAVE)
        self.spawn_wave(wave_size, precomputed_family=next_family)
        self.state.spawn_fixed_food()
        self.countdown = npd.ROUNDTIME
        self.rounds += 1
        self.round_start_frame = self.frame
        self.round_start_active_cells = self.active_cell_count()
        self.round_ended_early = False
        self.round_early_countdown_remaining = 0

    def advance_unrendered_round(self):
        while self.countdown > 0:
            if self.round_should_end_early():
                self.end_round_early()
                break
            step_count = min(self.args.tensor_block_steps, self.countdown)
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
            self.state.family_index.detach().cpu().numpy(),
            self.state.food_grid.detach().cpu().numpy(),
            self.state.npc_grid.detach().cpu().numpy(),
        )

    def render_round(self, writer, font):
        frames_written = 0
        while self.countdown > 0:
            if self.round_should_end_early():
                self.end_round_early()
                break
            index_grid, health, family_index, food_grid, npc_grid = self.snapshot()
            writer.append_data(render_tensor_frame(
                index_grid,
                health,
                family_index,
                food_grid,
                npc_grid,
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
        behavior_totals = self.behavior_totals_tensor.detach().cpu().to(torch.long).tolist()
        return {
            'active_cells_final': self.active_cell_count(),
            'action_device': str(self.device),
            'cell_capacity': self.state.cells,
            'cuda_graph_captures': self.cuda_graph_captures if self.device.type == 'cuda' else None,
            'cuda_name': torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else '',
            'elapsed_seconds': elapsed_seconds,
            'empty_refills': self.empty_refills,
            'early_ended_rounds': self.early_ended_rounds,
            'family_capacity_final': self.state.families,
            'food_per_round': int(npd.FOOD_PER_ROUND),
            'food_eaten_max_possible': int(npd.FOOD_PER_ROUND * self.rounds),
            'food_health_reward': float(npd.FOOD_HEALTH_REWARD),
            'food_remaining_final': int(self.state.food_grid.sum().item()),
            'frames_written': frames_written,
            'full_simulation_frames': self.frame,
            'fitness_update_lr': float(npd.FITNESS_UPDATE_LR),
            'kill_health_reward': float(npd.KILL_HEALTH_REWARD),
            'min_wave': int(npd.MIN_WAVE),
            'movement_health_cost': float(npd.MOVEMENT_HEALTH_COST),
            'npc_count': int(self.state.npc_flat_positions.numel()),
            'npc_input_value': float(npd.NPC_INPUT_VALUE),
            'npc_dodge_metric_summary': dodge_metric_summary(self.round_summaries),
            'per_wave': int(npd.PER_WAVE),
            'round_transition_health_cost': float(npd.ROUND_TRANSITION_HEALTH_COST),
            'roundtime': int(npd.ROUNDTIME),
            'stationary_damage_after_steps': int(npd.STATIONARY_DAMAGE_AFTER_STEPS),
            'stationary_health_cost': float(npd.STATIONARY_HEALTH_COST),
            'behavior_totals': {name: int(value) for name, value in zip(EVENT_COUNT_NAMES, behavior_totals)},
            'round_metrics_csv': str(Path(output).with_suffix(Path(output).suffix + '.round_metrics.csv')),
            'output': str(output),
            'rounds_completed': self.rounds,
            'rendered_rounds': rendered_rounds,
            'tensor_network_dtype': str(self.network_dtype).removeprefix('torch.'),
            'tensor_network_dtype_requested': self.args.tensor_network_dtype,
            'waves_spawned': self.waves_spawned,
        }

    def write_round_metrics(self, path):
        if not self.round_summaries:
            return
        fieldnames = list(self.round_summaries[0])
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.round_summaries)

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
                'early_ended_rounds': self.early_ended_rounds,
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
        f'metrics_only: {metrics["metrics_only"]}',
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
        f'tensor_network_dtype: {metrics["tensor_network_dtype"]}',
        f'tensor_network_dtype_requested: {metrics["tensor_network_dtype_requested"]}',
        f'tensor_coeff_scale: {args.tensor_coeff_scale}',
        f'tensor_stationary_health_cap: {args.tensor_stationary_health_cap}',
        f'tensor_compile_mode: {args.tensor_compile_mode}',
        f'tensor_matmul_precision: {args.tensor_matmul_precision}',
        f'fitness_update_lr: {metrics["fitness_update_lr"]}',
        f'movement_health_cost: {metrics["movement_health_cost"]}',
        f'stationary_health_cost: {metrics["stationary_health_cost"]}',
        f'stationary_damage_after_steps: {metrics["stationary_damage_after_steps"]}',
        f'round_transition_health_cost: {metrics["round_transition_health_cost"]}',
        f'roundtime: {metrics["roundtime"]}',
        f'per_wave: {metrics["per_wave"]}',
        f'min_wave: {metrics["min_wave"]}',
        f'npc_count: {metrics["npc_count"]}',
        f'npc_input_value: {metrics["npc_input_value"]}',
        f'food_per_round: {metrics["food_per_round"]}',
        f'food_eaten_max_possible: {metrics["food_eaten_max_possible"]}',
        f'food_health_reward: {metrics["food_health_reward"]}',
        f'kill_health_reward: {metrics["kill_health_reward"]}',
        f'food_remaining_final: {metrics["food_remaining_final"]}',
        f'cuda_graph_captures: {metrics["cuda_graph_captures"]}',
        f'family_capacity_final: {metrics["family_capacity_final"]}',
        f'active_cells_final: {metrics["active_cells_final"]}',
        f'waves_spawned: {metrics["waves_spawned"]}',
        f'final_state: {metrics.get("final_state", "")}',
        f'round_metrics_csv: {metrics["round_metrics_csv"]}',
        f'empty_refills: {metrics["empty_refills"]}',
        f'early_ended_rounds: {metrics["early_ended_rounds"]}',
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
    rounds_to_render = [] if args.metrics_only else selected_rounds(args.render_rounds, args.round_stride)
    rounds_to_render_set = set(rounds_to_render)
    rendered_rounds = []
    frames_written = 0
    font = ImageFont.load_default()

    started = time.perf_counter()
    with torch.inference_mode():
        run = TensorRank1VideoRun(args)
        if args.metrics_only:
            while run.rounds < args.rounds:
                if not run.advance_unrendered_round():
                    break
        else:
            with imageio.get_writer(output, fps=args.fps, macro_block_size=1) as writer:
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
        metrics['metrics_only'] = bool(args.metrics_only)
        run.write_round_metrics(Path(metrics['round_metrics_csv']))
        if args.save_final_state:
            state_path = run.save_state(args.save_final_state, metrics)
            metrics['final_state'] = str(state_path)

    write_manifest(output, args, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
