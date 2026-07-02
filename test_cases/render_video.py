#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import neural_petri_dish as npd


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('value must be positive')
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description='Render a Neural Petri Dish run to a video artifact.')
    parser.add_argument('--output', default='test_cases/artifacts/neural_petri_dish.mp4')
    parser.add_argument('--frames', type=positive_int, default=None)
    parser.add_argument('--fps', type=positive_int, default=24)
    parser.add_argument('--size', type=npd.parse_size, default=(18, 60), help='grid size as LINESxCOLUMNS')
    parser.add_argument('--initial-cells', type=positive_int, default=300)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--mutation-mode', choices=npd.MUTATION_MODES, default=npd.DEFAULT_MUTATION_MODE)
    parser.add_argument('--action-backend', choices=npd.ACTION_BACKENDS, default=npd.ACTION_BACKEND_SEQUENTIAL)
    parser.add_argument('--cell-size', type=positive_int, default=8)
    parser.add_argument('--status-height', type=positive_int, default=34)
    parser.add_argument(
        '--render-rounds',
        type=positive_int,
        default=None,
        help='render every visible frame for this many selected UI rounds',
    )
    parser.add_argument(
        '--round-stride',
        '--save-every-rounds',
        dest='round_stride',
        type=positive_int,
        default=1,
        help='gap between selected UI rounds when --render-rounds is used',
    )
    parser.add_argument('--write-manifest', action='store_true')
    args = parser.parse_args()
    if args.frames is not None and args.render_rounds is not None:
        parser.error('use either --frames or --render-rounds, not both')
    if args.frames is None and args.render_rounds is None:
        args.frames = 240
    return args


def xterm_256_to_rgb(color):
    if color < 16:
        palette = [
            (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
            (0, 0, 128), (128, 0, 128), (0, 128, 128), (192, 192, 192),
            (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0),
            (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
        ]
        return palette[color]
    if color < 232:
        color -= 16
        r = color // 36
        g = (color % 36) // 6
        b = color % 6
        levels = [0, 95, 135, 175, 215, 255]
        return levels[r], levels[g], levels[b]
    level = 8 + (color - 232) * 10
    return level, level, level


def status_text(game, countdown, frame):
    totalcells = len(game.cells)
    if totalcells:
        avghealth = round(sum(c.health for c in game.cells) / totalcells)
        maxhealth = max(c.health for c in game.cells)
        maxage = max(c.age for c in game.cells)
    else:
        avghealth = 0
        maxhealth = 0
        maxage = 0
    return (
        f'Frame {frame}  AVG HP {avghealth}  MAX HP {maxhealth}  '
        f'Cells {totalcells}  Oldest {maxage}  Rounds {game.rounds}  Countdown {countdown}'
    )


def render_frame(game, countdown, frame, cell_size, status_height, font):
    visible_rows = game.size.lines - 2
    width = game.size.columns * cell_size
    height = visible_rows * cell_size + status_height
    image = Image.new('RGB', (width, height), (8, 10, 14))
    draw = ImageDraw.Draw(image)

    for row in range(2, game.size.lines):
        for col in range(2, game.size.columns + 2):
            y0 = (row - 2) * cell_size
            x0 = (col - 2) * cell_size
            if game.grid[row, col] == 1:
                color = int((87 - 4 * np.sum(game.grid[row-1:row+2, col-1:col+2])) % 255)
                fill = xterm_256_to_rgb(color)
                draw.rectangle((x0, y0, x0 + cell_size - 1, y0 + cell_size - 1), fill=fill)
            elif cell_size >= 6:
                draw.point((x0, y0), fill=(16, 20, 26))

    status_y = visible_rows * cell_size
    draw.rectangle((0, status_y, width, height), fill=(14, 18, 24))
    draw.text((6, status_y + 9), status_text(game, countdown, frame), fill=(232, 238, 246), font=font)
    return np.asarray(image)


def ensure_cells(game):
    if not game.cells:
        game = npd.init(game, num=npd.MIN_WAVE)
    return game


def advance_update(game, countdown):
    # Mirror neural_petri_dish.main(): refill an empty dish before the tick,
    # then apply round transitions before each cell takes its action.
    game = ensure_cells(game)
    game, countdown = npd.advance_round(game, countdown)
    game = npd.step(game)
    return game, countdown - 1


def selected_rounds(render_rounds, round_stride):
    if render_rounds is None:
        return []
    return [round_index * round_stride for round_index in range(render_rounds)]


def record_round(rounds_rendered, round_num):
    if not rounds_rendered or rounds_rendered[-1] != round_num:
        rounds_rendered.append(round_num)


def write_manifest(path, args, frames_written, rounds_rendered):
    size = npd.terminal_size(args.size)
    frames = args.frames if args.render_rounds is None else 'off'
    render_rounds = args.render_rounds if args.render_rounds is not None else 'off'
    round_stride = args.round_stride if args.render_rounds is not None else 'off'
    manifest = [
        'Neural Petri Dish video artifact',
        '',
        f'output: {path}',
        f'frames: {frames}',
        f'render_rounds: {render_rounds}',
        f'round_stride: {round_stride}',
        f'frames_written: {frames_written}',
        f'fps: {args.fps}',
        f'size: {size.lines}x{size.columns}',
        f'initial_cells: {args.initial_cells}',
        f'seed: {args.seed}',
        f'mutation_mode: {args.mutation_mode}',
        f'action_backend: {args.action_backend}',
        f'cell_size: {args.cell_size}',
        f'rendered_rounds: {",".join(str(round_num) for round_num in rounds_rendered)}',
        '',
    ]
    path.with_suffix(path.suffix + '.manifest.txt').write_text('\n'.join(manifest), encoding='utf-8')


def write_fixed_frame_video(writer, game, countdown, args, font):
    frames_written = 0
    rounds_rendered = []
    for frame in range(args.frames):
        game = ensure_cells(game)
        writer.append_data(render_frame(game, countdown, frame, args.cell_size, args.status_height, font))
        frames_written += 1
        record_round(rounds_rendered, game.rounds)
        if frame + 1 < args.frames:
            game, countdown = advance_update(game, countdown)
    return frames_written, rounds_rendered


def write_selected_round_video(writer, game, countdown, args, font):
    rounds_to_render = selected_rounds(args.render_rounds, args.round_stride)
    rounds_to_render_set = set(rounds_to_render)
    last_round = rounds_to_render[-1]
    frame = 0
    rounds_rendered = []

    while game.rounds <= last_round:
        game = ensure_cells(game)
        # Selected-round previews still simulate every skipped round, but only
        # selected UI rounds are encoded into the artifact. This lets PR videos
        # compare later dynamics without hiding motion inside each kept round.
        if game.rounds in rounds_to_render_set:
            writer.append_data(render_frame(game, countdown, frame, args.cell_size, args.status_height, font))
            record_round(rounds_rendered, game.rounds)
            frame += 1
        game, countdown = advance_update(game, countdown)
    return frame, rounds_rendered


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    npd.seed_all(args.seed)
    with torch.no_grad():
        game = npd.init(
            npd.Game(size=args.size, mutation_mode=args.mutation_mode, action_backend=args.action_backend),
            num=args.initial_cells,
        )
        countdown = npd.ROUNDTIME
        with imageio.get_writer(output, fps=args.fps, macro_block_size=1) as writer:
            if args.render_rounds is None:
                frames_written, rounds_rendered = write_fixed_frame_video(writer, game, countdown, args, font)
            else:
                frames_written, rounds_rendered = write_selected_round_video(writer, game, countdown, args, font)

    if args.write_manifest:
        write_manifest(output, args, frames_written, rounds_rendered)
    print(f'wrote {frames_written} frames to {output}')


if __name__ == '__main__':
    main()
