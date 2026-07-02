#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import sys

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import neural_petri_dish as npd
import render_video


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('value must be positive')
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description='Record shared rank-1 mutation dynamics.')
    parser.add_argument('--output-dir', default='test_cases/artifacts/mutation_comparison')
    parser.add_argument('--rounds', type=positive_int, default=100)
    parser.add_argument('--render-every-rounds', type=positive_int, default=10)
    parser.add_argument('--fps', type=positive_int, default=30)
    parser.add_argument('--size', type=npd.parse_size, default=(18, 60), help='grid size as LINESxCOLUMNS')
    parser.add_argument('--initial-cells', type=positive_int, default=300)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--cell-size', type=positive_int, default=8)
    parser.add_argument('--status-height', type=positive_int, default=34)
    parser.add_argument('--roundtime', type=positive_int, default=npd.ROUNDTIME)
    parser.add_argument('--action-backend', choices=npd.ACTION_BACKENDS, default=npd.ACTION_BACKEND_SEQUENTIAL)
    parser.add_argument('--modes', nargs='+', choices=npd.MUTATION_MODES, default=list(npd.MUTATION_MODES))
    return parser.parse_args()


def ensure_cells(game):
    if not game.cells:
        return npd.init(game, num=npd.MIN_WAVE)
    return game


def selected_rounds(total_rounds, stride):
    return list(range(0, total_rounds, stride))


def cell_ids(game):
    return {id(cell) for cell in game.cells}


def health_summary(game):
    if not game.cells:
        return 0.0, 0, 0
    avg_health = sum(cell.health for cell in game.cells) / len(game.cells)
    max_health = max(cell.health for cell in game.cells)
    max_age = max(cell.age for cell in game.cells)
    return avg_health, max_health, max_age


def summarize_initial_round(mode, game):
    ids = cell_ids(game)
    avg_health, max_health, max_age = health_summary(game)
    row = {
        'mutation_mode': mode,
        'round': 0,
        'previous_round_cells': len(ids),
        'pre_refill_cells': len(ids),
        'post_refill_cells': len(ids),
        'survivors_from_previous_round': len(ids),
        'survival_rate': '1.000000',
        'avg_health': f'{avg_health:.3f}',
        'max_health': max_health,
        'max_age': max_age,
    }
    return row, ids


def summarize_round_transition(mode, round_num, previous_ids, pre_refill_ids, post_refill_game):
    post_refill_ids = cell_ids(post_refill_game)
    previous_count = len(previous_ids)
    survivors = len(pre_refill_ids & previous_ids)
    survival_rate = survivors / previous_count if previous_count else 0.0
    avg_health, max_health, max_age = health_summary(post_refill_game)

    row = {
        'mutation_mode': mode,
        'round': round_num,
        'previous_round_cells': previous_count,
        'pre_refill_cells': len(pre_refill_ids),
        'post_refill_cells': len(post_refill_ids),
        'survivors_from_previous_round': survivors,
        'survival_rate': f'{survival_rate:.6f}',
        'avg_health': f'{avg_health:.3f}',
        'max_health': max_health,
        'max_age': max_age,
    }
    return row, post_refill_ids


def write_video_manifest(path, args, mode, frames_written, rounds_rendered):
    size = npd.terminal_size(args.size)
    manifest = [
        'Neural Petri Dish mutation comparison video',
        '',
        f'output: {path}',
        f'mutation_mode: {mode}',
        f'rounds: {args.rounds}',
        f'render_every_rounds: {args.render_every_rounds}',
        f'fps: {args.fps}',
        f'size: {size.lines}x{size.columns}',
        f'initial_cells: {args.initial_cells}',
        f'seed: {args.seed}',
        f'roundtime: {args.roundtime}',
        f'action_backend: {args.action_backend}',
        f'frames_written: {frames_written}',
        f'rendered_rounds: {",".join(str(round_num) for round_num in rounds_rendered)}',
        '',
    ]
    path.with_suffix(path.suffix + '.manifest.txt').write_text('\n'.join(manifest), encoding='utf-8')


def run_mode(args, mode, output_dir):
    npd.seed_all(args.seed)
    npd.ROUNDTIME = args.roundtime

    game = npd.init(
        npd.Game(size=args.size, mutation_mode=mode, action_backend=args.action_backend),
        num=args.initial_cells,
    )
    countdown = npd.ROUNDTIME
    font = ImageFont.load_default()
    rounds_to_render = set(selected_rounds(args.rounds, args.render_every_rounds))
    rendered_rounds = []
    frames_written = 0
    metrics = []

    row, previous_ids = summarize_initial_round(mode, game)
    metrics.append(row)

    video_path = output_dir / f'{mode}_rounds_{args.rounds}_every_{args.render_every_rounds}.mp4'
    with torch.inference_mode(), imageio.get_writer(video_path, fps=args.fps, macro_block_size=1) as writer:
        while game.rounds < args.rounds:
            game = ensure_cells(game)
            if game.rounds in rounds_to_render:
                writer.append_data(
                    render_video.render_frame(
                        game,
                        countdown,
                        frames_written,
                        args.cell_size,
                        args.status_height,
                        font,
                    )
                )
                if not rendered_rounds or rendered_rounds[-1] != game.rounds:
                    rendered_rounds.append(game.rounds)
                frames_written += 1

            pre_refill_ids = cell_ids(game) if countdown == 0 else None
            previous_round = game.rounds
            game, countdown = npd.advance_round(game, countdown)
            if game.rounds != previous_round:
                row, previous_ids = summarize_round_transition(
                    mode,
                    game.rounds,
                    previous_ids,
                    pre_refill_ids,
                    game,
                )
                metrics.append(row)

            game = npd.step(game)
            countdown -= 1

    write_video_manifest(video_path, args, mode, frames_written, rendered_rounds)
    return metrics


def write_csv(path, rows):
    fieldnames = [
        'mutation_mode',
        'round',
        'previous_round_cells',
        'pre_refill_cells',
        'post_refill_cells',
        'survivors_from_previous_round',
        'survival_rate',
        'avg_health',
        'max_health',
        'max_age',
    ]
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def by_mode(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['mutation_mode'], []).append(row)
    return grouped


def nice_ticks(max_round):
    step = max(1, max_round // 5)
    ticks = list(range(0, max_round + 1, step))
    if ticks[-1] != max_round:
        ticks.append(max_round)
    return ticks


def draw_panel(draw, box, rows_by_mode, metric, title, y_min, y_max, colors, font, max_round):
    left, top, right, bottom = box
    draw.rectangle(box, outline=(80, 88, 98), width=1)
    draw.text((left, top - 22), title, fill=(28, 32, 38), font=font)

    for tick in nice_ticks(max_round):
        x = left + (right - left) * tick / max_round
        draw.line((x, bottom, x, bottom + 5), fill=(80, 88, 98))
        draw.text((x - 10, bottom + 8), str(tick), fill=(80, 88, 98), font=font)

    for fraction, label in [(0.0, f'{y_min:.2f}'), (0.5, f'{(y_min + y_max) / 2:.2f}'), (1.0, f'{y_max:.2f}')]:
        y = bottom - (bottom - top) * fraction
        draw.line((left - 5, y, left, y), fill=(80, 88, 98))
        draw.text((8, y - 6), label, fill=(80, 88, 98), font=font)

    y_span = max(y_max - y_min, 1e-9)
    for mode, mode_rows in rows_by_mode.items():
        points = []
        for row in mode_rows:
            x_value = int(row['round'])
            y_value = float(row[metric])
            x = left + (right - left) * x_value / max_round
            y = bottom - (bottom - top) * (y_value - y_min) / y_span
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=colors[mode], width=3)


def write_plot(path, rows, rounds):
    plot_rows = [row for row in rows if int(row['round']) > 0] or rows
    rows_by_mode = by_mode(plot_rows)
    colors = {npd.MUTATION_MODE_SHARED_RANK1_FACTORED: (130, 86, 180)}
    image = Image.new('RGB', (1200, 760), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((70, 24), 'Mutation mode comparison after initial spawn', fill=(20, 24, 30), font=font)
    legend_x = 880
    for index, mode in enumerate(rows_by_mode):
        y = 26 + index * 18
        draw.line((legend_x, y + 6, legend_x + 28, y + 6), fill=colors[mode], width=4)
        draw.text((legend_x + 36, y), mode, fill=(20, 24, 30), font=font)

    max_survivors = max(int(row['survivors_from_previous_round']) for row in plot_rows)
    draw_panel(
        draw,
        (70, 90, 1160, 350),
        rows_by_mode,
        'survivors_from_previous_round',
        'Cells from previous round still alive before refill',
        0.0,
        max(max_survivors * 1.05, 1.0),
        colors,
        font,
        rounds,
    )
    draw_panel(
        draw,
        (70, 440, 1160, 700),
        rows_by_mode,
        'survival_rate',
        'Fraction of previous round cells still alive',
        0.0,
        1.0,
        colors,
        font,
        rounds,
    )
    draw.text((580, 732), 'round', fill=(80, 88, 98), font=font)
    image.save(path)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode in args.modes:
        rows.extend(run_mode(args, mode, output_dir))

    csv_path = output_dir / f'survival_rounds_{args.rounds}.csv'
    plot_path = output_dir / f'survival_rounds_{args.rounds}.png'
    write_csv(csv_path, rows)
    write_plot(plot_path, rows, args.rounds)

    print(f'wrote comparison CSV to {csv_path}')
    print(f'wrote comparison plot to {plot_path}')


if __name__ == '__main__':
    main()
