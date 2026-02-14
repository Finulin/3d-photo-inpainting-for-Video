#!/usr/bin/env python3
"""
stereo_from_depth_video.py

Generate left/right stereo videos from an input RGB video and corresponding depth maps.

Default output: MP4 (H.264) using libx264 in near-lossless mode (-crf 0) with yuv444p to avoid chroma subsampling.
FFmpeg is required and must be available in PATH.

Usage example:
  python3 stereo_from_depth_video.py --video video/moon_circle.mp4 --depth depth_frames_dir --out_dir out_vid --max_shift 40

Depth input can be either:
- a video file (same number of frames as color video preferred), or
- a directory containing per-frame depth files (.npy or image files) in sorted order.

The script writes intermediate PNG frames and then encodes two videos with ffmpeg.
"""
import argparse
import sys
import shutil
import subprocess
from pathlib import Path
import cv2
import numpy as np

try:
    from stereo_from_depth import generate_stereo_pair
except Exception as e:
    print("Error importing generate_stereo_pair from stereo_from_depth.py:", e)
    raise


def is_video_file(p: str):
    p = p.lower()
    return any(p.endswith(ext) for ext in ('.mp4', '.mov', '.avi', '.mkv', '.webm'))


def sorted_frame_files(d: Path):
    exts = ['.npy', '.png', '.jpg', '.jpeg', '.tiff', '.tif']
    files = [f for f in sorted(d.iterdir()) if f.suffix.lower() in exts]
    return files


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def build_ffmpeg_cmd(frames_pattern: str, fps: float, out_path: str):
    # Use libx264 lossless setting (-crf 0) and yuv444p to avoid chroma subsampling.
    # This produces large files but preserves color information best for H.264 in MP4.
    return [
        'ffmpeg', '-y', '-framerate', str(fps), '-i', frames_pattern,
        '-c:v', 'libx264', '-preset', 'veryslow', '-crf', '0', '-pix_fmt', 'yuv444p', out_path
    ]


def main():
    parser = argparse.ArgumentParser(description='Generate left/right stereo videos from video + depth frames')
    parser.add_argument('--video', required=True, help='Input RGB video file')
    parser.add_argument('--depth', required=True, help='Depth source: video file or directory with per-frame depth (.npy/.png)')
    parser.add_argument('--out_dir', default='out_vid', help='Output directory')
    parser.add_argument('--max_shift', type=float, default=40.0, help='Maximum pixel shift for closest points')
    parser.add_argument('--fill_method', choices=['inpaint', 'morph'], default='inpaint', help='Hole filling method')
    parser.add_argument('--keep_frames', action='store_true', help='Do not delete intermediate frames')
    parser.add_argument('--pipe', action='store_true', help='Stream frames to ffmpeg via stdin (no intermediate PNG files)')
    args = parser.parse_args()

    if shutil.which('ffmpeg') is None:
        print('Error: ffmpeg not found in PATH. Please install ffmpeg and ensure it is on your PATH.')
        sys.exit(1)

    video_path = Path(args.video)
    depth_src = Path(args.depth)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    left_frames = out_dir / 'left_frames'
    right_frames = out_dir / 'right_frames'
    if not args.pipe:
        ensure_dir(left_frames)
        ensure_dir(right_frames)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print('Error: cannot open video', video_path)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    depth_is_video = False
    depth_cap = None
    depth_files = []
    if is_video_file(str(depth_src)):
        depth_is_video = True
        depth_cap = cv2.VideoCapture(str(depth_src))
        if not depth_cap.isOpened():
            print('Error: cannot open depth video', depth_src)
            sys.exit(1)
        depth_frame_count = int(depth_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if depth_frame_count != 0 and total_frames != 0 and depth_frame_count != total_frames:
            print(f'Warning: color video has {total_frames} frames, depth video has {depth_frame_count} frames')
    else:
        if not depth_src.exists() or not depth_src.is_dir():
            print('Error: depth must be a video file or an existing directory of per-frame depth files')
            sys.exit(1)
        depth_files = sorted_frame_files(depth_src)
        if total_frames != 0 and len(depth_files) != 0 and len(depth_files) != total_frames:
            print(f'Warning: color video has {total_frames} frames, depth folder has {len(depth_files)} files')

    frame_idx = 0
    saved = 0

    # If piping, start ffmpeg processes that read PNG from stdin
    if args.pipe:
        out_left = out_dir / 'left.mp4'
        out_right = out_dir / 'right.mp4'
        left_cmd = [
            'ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(fps), '-i', '-',
            '-c:v', 'libx264', '-preset', 'veryslow', '-crf', '0', '-pix_fmt', 'yuv444p', str(out_left)
        ]
        right_cmd = [
            'ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(fps), '-i', '-',
            '-c:v', 'libx264', '-preset', 'veryslow', '-crf', '0', '-pix_fmt', 'yuv444p', str(out_right)
        ]
        print('Starting ffmpeg processes (pipe mode)...')
        left_proc = subprocess.Popen(left_cmd, stdin=subprocess.PIPE)
        right_proc = subprocess.Popen(right_cmd, stdin=subprocess.PIPE)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if depth_is_video:
                dret, dframe = depth_cap.read()
                if not dret:
                    print(f'Warning: depth video ended at frame {frame_idx}')
                    depth = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.float32)
                else:
                    if dframe.ndim == 3:
                        dgray = cv2.cvtColor(dframe, cv2.COLOR_BGR2GRAY)
                    else:
                        dgray = dframe
                    depth = dgray.astype(np.float32)
            else:
                if frame_idx < len(depth_files):
                    df = depth_files[frame_idx]
                    if df.suffix.lower() == '.npy':
                        depth = np.load(str(df)).astype(np.float32)
                    else:
                        dimg = cv2.imread(str(df), cv2.IMREAD_UNCHANGED)
                        if dimg is None:
                            print(f'Warning: cannot read depth file {df}; using zeros')
                            depth = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.float32)
                        else:
                            if dimg.ndim == 3:
                                dimg = cv2.cvtColor(dimg, cv2.COLOR_BGR2GRAY)
                            depth = dimg.astype(np.float32)
                else:
                    print(f'Warning: missing depth file for frame {frame_idx}; using zeros')
                    depth = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.float32)

            if depth.shape[0] != img_rgb.shape[0] or depth.shape[1] != img_rgb.shape[1]:
                depth = cv2.resize(depth, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

            left, right = generate_stereo_pair(img_rgb, depth, max_shift_pixels=args.max_shift, fill_method=args.fill_method)

            left_bgr = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
            right_bgr = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)

            if args.pipe:
                # encode PNG to memory and write to ffmpeg stdin
                ok_l, buf_l = cv2.imencode('.png', left_bgr)
                ok_r, buf_r = cv2.imencode('.png', right_bgr)
                if not ok_l or not ok_r:
                    print(f'Error encoding frame {frame_idx} to PNG; skipping')
                else:
                    left_proc.stdin.write(buf_l.tobytes())
                    right_proc.stdin.write(buf_r.tobytes())
            else:
                left_path = left_frames / f'left_{frame_idx:06d}.png'
                right_path = right_frames / f'right_{frame_idx:06d}.png'
                cv2.imwrite(str(left_path), left_bgr)
                cv2.imwrite(str(right_path), right_bgr)

            frame_idx += 1
            saved += 1
            if frame_idx % 50 == 0:
                print(f'Processed frames: {frame_idx}')

    finally:
        cap.release()
        if depth_cap is not None:
            depth_cap.release()

    if saved == 0:
        print('No frames processed; aborting ffmpeg step.')
        sys.exit(1)

    if args.pipe:
        # finish pipe writing and wait for ffmpeg to finish
        left_proc.stdin.close()
        right_proc.stdin.close()
        left_proc.wait()
        right_proc.wait()
    else:
        left_pattern = str(left_frames / 'left_%06d.png')
        right_pattern = str(right_frames / 'right_%06d.png')

        out_left = out_dir / 'left.mp4'
        out_right = out_dir / 'right.mp4'

        cmd_left = build_ffmpeg_cmd(left_pattern, fps, str(out_left))
        cmd_right = build_ffmpeg_cmd(right_pattern, fps, str(out_right))

        print('Encoding left video with ffmpeg...')
        print(' '.join(cmd_left))
        subprocess.check_call(cmd_left)

        print('Encoding right video with ffmpeg...')
        print(' '.join(cmd_right))
        subprocess.check_call(cmd_right)

    print('Wrote:')
    print('  ', out_left)
    print('  ', out_right)

    if not args.keep_frames and not args.pipe:
        print('Removing intermediate frames...')
        shutil.rmtree(left_frames)
        shutil.rmtree(right_frames)


if __name__ == '__main__':
    main()
