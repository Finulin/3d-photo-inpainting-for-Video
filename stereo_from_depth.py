#!/usr/bin/env python3
"""
Generate a left/right stereo pair from a single RGB image and a depth map.

This is a lightweight utility: it computes a per-pixel horizontal shift from
an inverse-depth proxy and warps the image to create left/right eye views.

Usage (example):
    python3 stereo_from_depth.py --image image/moon.jpg --depth depth/moon.npy --out_dir out --max_shift 40

Defaults are conservative; tune --max_shift (pixels) to increase/decrease stereo
separation. Holes caused by the warp are filled using OpenCV inpainting.
"""
import os
import argparse
import numpy as np
import cv2


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {path}")
    # convert BGR->RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_depth(path):
    # Support .npy or image formats (png)
    if path.lower().endswith('.npy'):
        depth = np.load(path)
    else:
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(f"Depth map not found or cannot be read: {path}")
        # If loaded as 3-channel, convert to grayscale
        if d.ndim == 3:
            d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        depth = d.astype(np.float32)

    # Ensure float32
    depth = depth.astype(np.float32)
    # If depth seems to be in 0..255 (uint8), scale to 0..1
    if depth.max() > 0 and depth.max() <= 255 and depth.dtype == np.float32:
        # But don't blindly rescale if values are already floats >1 (pfm-like)
        depth = depth / 255.0

    return depth


def inv_depth_normalized(depth, eps=1e-6, clip_percentiles=(1.0, 99.0)):
    """Return a normalized inverse-depth map in range [0,1].

    depth: float32 array where larger values are farther (typical). Zeros are treated as unknown and set to 0.
    """
    mask = depth > eps
    inv = np.zeros_like(depth, dtype=np.float32)
    if mask.any():
        inv[mask] = 1.0 / (depth[mask] + eps)
        # clip outliers
        pmin, pmax = np.percentile(inv[mask], clip_percentiles)
        if pmax - pmin > 1e-6:
            inv[mask] = np.clip(inv[mask], pmin, pmax)
        # normalize to [0,1]
        mn = inv[mask].min()
        mx = inv[mask].max()
        if mx - mn > 1e-6:
            inv[mask] = (inv[mask] - mn) / (mx - mn)
        else:
            inv[mask] = 0.0

    return inv


def warp_horizontal(img, shift_map, fill_method='inpaint'):
    """Warp an RGB image horizontally according to per-pixel shift_map (pixels).

    shift_map: float32 array same HxW, positive means shift to the right.
    Returns warped image with holes filled.
    """
    H, W = shift_map.shape
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    map_x = (xs + shift_map).astype(np.float32)
    map_y = ys.astype(np.float32)

    # img is RGB uint8
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # find holes: places that became black (0,0,0). This may also pick up real black pixels,
    # but that's acceptable for most photos. We restrict to areas where original image was not black.
    hole_mask = np.all(warped == 0, axis=2).astype(np.uint8)

    if hole_mask.max() == 0:
        return warped

    if fill_method == 'inpaint':
        # inpaint requires BGR and 8-bit single-channel mask
        warped_bgr = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
        inpainted = cv2.inpaint(warped_bgr, hole_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    else:
        # simple morphological close to reduce holes
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, kernel, iterations=2)
        return closed


def generate_stereo_pair(img, depth, max_shift_pixels=40, fill_method='inpaint'):
    """Generate (left, right) views from center image + depth.

    - img: HxWx3 uint8 RGB
    - depth: HxW float32 array (arbitrary scale)
    - max_shift_pixels: maximum horizontal disparity (in pixels) for the closest points

    Returns left_img, right_img as uint8 RGB arrays.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError('img must be HxWx3 uint8 RGB')

    H, W = depth.shape
    if img.shape[0] != H or img.shape[1] != W:
        # resize depth to image
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    invn = inv_depth_normalized(depth)
    # positive shifts for closer (invn near 1)
    shift = invn * float(max_shift_pixels)

    # left eye: shift image right for close objects (i.e., viewer's left sees scene shifted right)
    half = shift * 0.5
    left = warp_horizontal(img, +half, fill_method=fill_method)
    right = warp_horizontal(img, -half, fill_method=fill_method)

    return left, right


def save_rgb(path, img_rgb):
    # save RGB as PNG (cv2.imwrite expects BGR)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def main():
    parser = argparse.ArgumentParser(description='Generate stereo pair from image + depth')
    parser.add_argument('--image', required=True, help='Path to RGB image')
    parser.add_argument('--depth', required=True, help='Path to depth map (.npy or image)')
    parser.add_argument('--out_dir', default='out', help='Output directory')
    parser.add_argument('--max_shift', type=float, default=40.0, help='Max pixel shift (closest point)')
    parser.add_argument('--fill_method', choices=['inpaint', 'morph'], default='inpaint', help='Hole filling method')
    args = parser.parse_args()

    img = load_image(args.image)
    depth = load_depth(args.depth)

    left, right = generate_stereo_pair(img, depth, max_shift_pixels=args.max_shift, fill_method=args.fill_method)

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]
    left_path = os.path.join(args.out_dir, base + '_left.png')
    right_path = os.path.join(args.out_dir, base + '_right.png')
    save_rgb(left_path, left)
    save_rgb(right_path, right)

    print(f'Wrote: {left_path}\n      {right_path}')


if __name__ == '__main__':
    main()
