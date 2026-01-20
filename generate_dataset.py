#!/usr/bin/env python3
"""
generate_dataset.py — Whole-window chatbot detector dataset generator (YOLOv8)

What it does
- Creates synthetic "desktop context" images by pasting FULL chatbot windows onto backgrounds.
- Also creates hard negatives by pasting FULL negative windows (Discord/Chrome/OBS/Google/etc.) with EMPTY labels.
- Also creates pure background negatives (no pasted window) with EMPTY labels.
- Writes a YOLOv8-ready dataset:
    out/
      images/train, images/val
      labels/train, labels/val
      data.yaml

Expected assets layout (matches your zip):
assets/
  bg/
    *.png|jpg|webp
  fg/
    chatGPT/
    Gemini/
    Claude/
    Negatives/

Class IDs:
0 chatgpt
1 gemini
2 claude

IMPORTANT: This script does NOT generate any random grey occlusion boxes.

Typical best-practice run (single run, strong negatives, wide scale range):
python generate_dataset.py --assets assets --out dataset --n 8000 --val-ratio 0.2 \
  --p-positive 0.60 --p-negative-window 0.85 \
  --max-overlays 1 --allow-offscreen --offscreen-frac 0.10 \
  --min-rel-w-win 0.25 --max-rel-w-win 0.75 \
  --min-rel-w-full 0.80 --max-rel-w-full 1.10 \
  --bg-jitter 0.30 --fg-jitter 0.40 --twist-strength 0.15 --blur-prob 0.08 --jpeg-prob 0.35

Train (example):
yolo detect train data=dataset/data.yaml model=yolov8n.pt imgsz=960 epochs=50 batch=8
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

CLASS_NAMES = ["chatgpt", "gemini", "claude"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}

# Allow common casing variants in your assets/fg folder
FG_CLASS_DIR_CANDIDATES = {
    "chatgpt": ["chatGPT", "ChatGPT", "chatgpt"],
    "gemini": ["Gemini", "gemini"],
    "claude": ["Claude", "claude"],
}
NEG_DIR_CANDIDATES = ["Negatives", "negatives", "NEGATIVES"]


# -----------------------------
# Utilities
# -----------------------------

def list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def maybe_downscale(im: Image.Image, limit: int) -> Image.Image:
    w, h = im.size
    m = max(w, h)
    if m <= limit:
        return im
    scale = limit / m
    return im.resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.Resampling.LANCZOS)


def to_rgb(im: Image.Image) -> Image.Image:
    return im if im.mode == "RGB" else im.convert("RGB")


def to_rgba(im: Image.Image) -> Image.Image:
    return im if im.mode == "RGBA" else im.convert("RGBA")


def resolve_dir(root: Path, candidates: List[str]) -> Optional[Path]:
    # direct
    for name in candidates:
        p = root / name
        if p.exists() and p.is_dir():
            return p
    # case-insensitive fallback
    cand_lower = {c.lower() for c in candidates}
    if root.exists():
        for p in root.iterdir():
            if p.is_dir() and p.name.lower() in cand_lower:
                return p
    return None


# -----------------------------
# Augmentations (photometric)
# -----------------------------

def photometric_jitter(im_rgb: Image.Image, strength: float) -> Image.Image:
    """Brightness/contrast/color jitter. strength in [0,1]."""
    if strength <= 0:
        return im_rgb
    im_rgb = ImageEnhance.Brightness(im_rgb).enhance(random.uniform(1 - 0.35 * strength, 1 + 0.35 * strength))
    im_rgb = ImageEnhance.Contrast(im_rgb).enhance(random.uniform(1 - 0.35 * strength, 1 + 0.35 * strength))
    im_rgb = ImageEnhance.Color(im_rgb).enhance(random.uniform(1 - 0.55 * strength, 1 + 0.55 * strength))
    return im_rgb


def maybe_jpeg_artifacts(im_rgb: Image.Image, p: float) -> Image.Image:
    """Simulate recompression artifacts."""
    if random.random() >= p:
        return im_rgb
    buf = BytesIO()
    q = random.randint(35, 90)
    im_rgb.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# -----------------------------
# Augmentations (geometry)
# -----------------------------

def _find_perspective_coeffs(dst_quad, src_quad) -> Tuple[float, ...]:
    """
    Compute PIL perspective coefficients mapping dst->src.
    dst_quad: [(u,v)x4] destination quad in output image
    src_quad: [(x,y)x4] source quad in input image
    """
    A = []
    B = []
    for (u, v), (x, y) in zip(dst_quad, src_quad):
        A.append([u, v, 1, 0, 0, 0, -x * u, -x * v])
        B.append(x)
        A.append([0, 0, 0, u, v, 1, -y * u, -y * v])
        B.append(y)

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    coeffs = np.linalg.lstsq(A, B, rcond=None)[0]
    return tuple(coeffs.tolist())


def mild_twist(rgba: Image.Image, strength: float) -> Image.Image:
    """Mild perspective warp; strength in [0,1]."""
    if strength <= 0 or random.random() > strength:
        return rgba

    w, h = rgba.size
    max_dx = int(w * random.uniform(0.01, 0.03) * strength)
    max_dy = int(h * random.uniform(0.01, 0.03) * strength)

    dst = [
        (0 + random.randint(-max_dx, max_dx), 0 + random.randint(-max_dy, max_dy)),  # TL
        (w + random.randint(-max_dx, max_dx), 0 + random.randint(-max_dy, max_dy)),  # TR
        (w + random.randint(-max_dx, max_dx), h + random.randint(-max_dy, max_dy)),  # BR
        (0 + random.randint(-max_dx, max_dx), h + random.randint(-max_dy, max_dy)),  # BL
    ]

    xs = [p[0] for p in dst]
    ys = [p[1] for p in dst]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    out_w = max(8, max_x - min_x)
    out_h = max(8, max_y - min_y)

    dst_shifted = [(x - min_x, y - min_y) for (x, y) in dst]
    src = [(0, 0), (w, 0), (w, h), (0, h)]

    coeffs = _find_perspective_coeffs(dst_shifted, src)
    return rgba.transform(
        (out_w, out_h),
        Image.Transform.PERSPECTIVE,
        coeffs,
        resample=Image.Resampling.BICUBIC,
    )


def alpha_tight_bbox(rgba: Image.Image, alpha_thresh: int = 8) -> Optional[Tuple[int, int, int, int]]:
    """Crop away transparent padding if present."""
    if rgba.mode != "RGBA":
        w, h = rgba.size
        return (0, 0, w, h)
    alpha = rgba.split()[-1]
    mask = alpha.point(lambda a: 255 if a > alpha_thresh else 0)
    return mask.getbbox()


def transform_foreground(
    fg: Image.Image,
    bg_w: int,
    bg_h: int,
    min_rel_w: float,
    max_rel_w: float,
    fg_jitter: float,
    twist_strength: float,
    blur_prob: float,
) -> Image.Image:
    """
    Whole-window transforms:
    - photometric jitter
    - scale relative to bg width (min_rel_w..max_rel_w)
    - mild perspective twist
    - optional blur
    - remove transparent padding if any
    """
    fg = to_rgba(fg)

    if fg_jitter > 0:
        rgb = photometric_jitter(fg.convert("RGB"), fg_jitter)
        fg = Image.merge("RGBA", (*rgb.split(), fg.split()[-1]))

    target_w = random.uniform(min_rel_w, max_rel_w) * bg_w
    scale = target_w / max(1, fg.size[0])
    new_w = max(8, int(fg.size[0] * scale))
    new_h = max(8, int(fg.size[1] * scale))
    fg = fg.resize((new_w, new_h), Image.Resampling.LANCZOS)

    fg = mild_twist(fg, twist_strength)

    if random.random() < blur_prob:
        fg = fg.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))

    bbox = alpha_tight_bbox(fg)
    if bbox:
        fg = fg.crop(bbox)

    return fg


def paste_random(
    bg_rgb: Image.Image,
    fg_rgba: Image.Image,
    allow_offscreen: bool,
    offscreen_frac: float,
    quantize_xy: int,
) -> Tuple[Image.Image, Optional[Tuple[int, int, int, int]]]:
    """
    Paste fg onto bg at random x,y. If allow_offscreen, fg can be partially out of frame.
    Robust to fg bigger than bg: clamps ranges and/or centers when necessary.
    """
    bg_w, bg_h = bg_rgb.size
    fg_w, fg_h = fg_rgba.size

    if allow_offscreen:
        max_off_x = int(offscreen_frac * fg_w)
        max_off_y = int(offscreen_frac * fg_h)

        x_low = -max_off_x
        x_high = bg_w - fg_w + max_off_x
        y_low = -max_off_y
        y_high = bg_h - fg_h + max_off_y

        # If the allowable range is invalid (fg too large), center it instead of crashing
        if x_high < x_low:
            x0 = (bg_w - fg_w) // 2
        else:
            x0 = random.randint(x_low, x_high)

        if y_high < y_low:
            y0 = (bg_h - fg_h) // 2
        else:
            y0 = random.randint(y_low, y_high)

    else:
        # Ensure it fits
        if fg_w >= bg_w or fg_h >= bg_h:
            shrink = min((bg_w - 2) / fg_w, (bg_h - 2) / fg_h, 0.9)
            fg_rgba = fg_rgba.resize(
                (max(8, int(fg_w * shrink)), max(8, int(fg_h * shrink))),
                Image.Resampling.LANCZOS,
            )
            fg_w, fg_h = fg_rgba.size
        x0 = random.randint(0, bg_w - fg_w)
        y0 = random.randint(0, bg_h - fg_h)

    if quantize_xy > 1:
        x0 = (x0 // quantize_xy) * quantize_xy
        y0 = (y0 // quantize_xy) * quantize_xy

    vis_x1 = max(0, x0)
    vis_y1 = max(0, y0)
    vis_x2 = min(bg_w, x0 + fg_w)
    vis_y2 = min(bg_h, y0 + fg_h)
    if vis_x2 <= vis_x1 or vis_y2 <= vis_y1:
        return bg_rgb, None

    bg = bg_rgb.convert("RGBA")
    tmp = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    tmp.alpha_composite(fg_rgba, (x0, y0))
    out = Image.alpha_composite(bg, tmp).convert("RGB")
    return out, (vis_x1, vis_y1, vis_x2, vis_y2)


# -----------------------------
# YOLO label writing
# -----------------------------

def yolo_line(class_id: int, bbox_xyxy: Tuple[int, int, int, int], img_w: int, img_h: int) -> str:
    x1, y1, x2, y2 = bbox_xyxy
    xc = ((x1 + x2) / 2.0) / img_w
    yc = ((y1 + y2) / 2.0) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    xc, yc, bw, bh = (clamp(xc, 0, 1), clamp(yc, 0, 1), clamp(bw, 0, 1), clamp(bh, 0, 1))
    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def write_data_yaml(out_root: Path) -> None:
    names_lines = "\n".join([f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)])
    (out_root / "data.yaml").write_text(
        f"path: {out_root.name}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"names:\n{names_lines}\n",
        encoding="utf-8",
    )


# -----------------------------
# Main
# -----------------------------

@dataclass
class Pools:
    bg: List[Path]
    fg_pos: Dict[str, List[Path]]  # chatgpt/gemini/claude
    fg_neg: List[Path]


def load_pools(assets: Path) -> Pools:
    bg_dir = assets / "bg"
    fg_root = assets / "fg"

    bg = list_images(bg_dir)
    if not bg:
        raise SystemExit(f"No backgrounds found in: {bg_dir.resolve()}")

    fg_pos: Dict[str, List[Path]] = {}
    for cname in CLASS_NAMES:
        cls_dir = resolve_dir(fg_root, FG_CLASS_DIR_CANDIDATES[cname])
        if cls_dir is None:
            raise SystemExit(f"Missing fg class folder for {cname} under: {fg_root.resolve()}")
        paths = list_images(cls_dir)
        if not paths:
            raise SystemExit(f"No foregrounds found in: {cls_dir.resolve()}")
        fg_pos[cname] = paths

    neg_dir = resolve_dir(fg_root, NEG_DIR_CANDIDATES)
    fg_neg = list_images(neg_dir) if neg_dir else []
    return Pools(bg=bg, fg_pos=fg_pos, fg_neg=fg_neg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=str, default="assets")
    ap.add_argument("--out", type=str, default="dataset")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--img-limit", type=int, default=1920)
    ap.add_argument("--seed", type=int, default=42)

    # Mix controls
    ap.add_argument("--p-positive", type=float, default=0.60,
                    help="Probability to generate a positive image (>=1 chatbot pasted)")
    ap.add_argument("--p-negative-window", type=float, default=0.85,
                    help="When generating a negative image, probability to paste a negative window. Otherwise background-only.")
    ap.add_argument("--max-overlays", type=int, default=1,
                    help="Max number of chatbot windows pasted per positive image (recommend 1).")

    # Placement
    ap.add_argument("--allow-offscreen", action="store_true")
    ap.add_argument("--offscreen-frac", type=float, default=0.10)
    ap.add_argument("--quantize-xy", type=int, default=2)

    # Scale ranges for fullscreen/windowed (based on filename containing 'fullscreen')
    ap.add_argument("--min-rel-w-win", type=float, default=0.25)
    ap.add_argument("--max-rel-w-win", type=float, default=0.75)
    ap.add_argument("--min-rel-w-full", type=float, default=0.80)
    ap.add_argument("--max-rel-w-full", type=float, default=1.10)

    # Augmentations
    ap.add_argument("--bg-jitter", type=float, default=0.30)
    ap.add_argument("--fg-jitter", type=float, default=0.40)
    ap.add_argument("--twist-strength", type=float, default=0.15)
    ap.add_argument("--blur-prob", type=float, default=0.08)
    ap.add_argument("--jpeg-prob", type=float, default=0.35)

    # Progress
    ap.add_argument("--print-every", type=int, default=200, help="Print progress every N images")

    args = ap.parse_args()
    random.seed(args.seed)

    assets = Path(args.assets)
    out_root = Path(args.out)

    pools = load_pools(assets)

    if args.p_negative_window > 0 and not pools.fg_neg:
        print("Warning: No assets/fg/Negatives found; negatives will be background-only.", flush=True)

    # Prepare output dirs
    for split in ("train", "val"):
        ensure_dir(out_root / "images" / split)
        ensure_dir(out_root / "labels" / split)

    # Split indices
    idxs = list(range(args.n))
    random.shuffle(idxs)
    val_n = int(args.n * args.val_ratio)
    val_set = set(idxs[:val_n])

    # Generation loop
    for i in range(args.n):
        split = "val" if i in val_set else "train"

        bg_path = random.choice(pools.bg)
        bg = Image.open(bg_path).convert("RGB")
        bg = maybe_downscale(bg, args.img_limit)
        bg = photometric_jitter(bg, args.bg_jitter)
        bg_w, bg_h = bg.size

        labels: List[str] = []

        # Decide positive vs negative
        if random.random() < args.p_positive:
            n_over = random.randint(1, args.max_overlays)
            for _ in range(n_over):
                cname = random.choice(CLASS_NAMES)
                fg_path = random.choice(pools.fg_pos[cname])

                fname = fg_path.name.lower()
                if "fullscreen" in fname:
                    min_rel_w, max_rel_w = args.min_rel_w_full, args.max_rel_w_full
                else:
                    min_rel_w, max_rel_w = args.min_rel_w_win, args.max_rel_w_win

                fg = Image.open(fg_path)
                fg_t = transform_foreground(
                    fg=fg,
                    bg_w=bg_w,
                    bg_h=bg_h,
                    min_rel_w=min_rel_w,
                    max_rel_w=max_rel_w,
                    fg_jitter=args.fg_jitter,
                    twist_strength=args.twist_strength,
                    blur_prob=args.blur_prob,
                )

                bg, bbox = paste_random(
                    bg_rgb=bg,
                    fg_rgba=fg_t,
                    allow_offscreen=args.allow_offscreen,
                    offscreen_frac=args.offscreen_frac,
                    quantize_xy=args.quantize_xy,
                )

                if bbox is not None:
                    labels.append(yolo_line(CLASS_TO_ID[cname], bbox, bg_w, bg_h))

        else:
            # Negative image: either paste a negative window or keep background-only
            if pools.fg_neg and random.random() < args.p_negative_window:
                neg_path = random.choice(pools.fg_neg)
                fname = neg_path.name.lower()
                if "fullscreen" in fname:
                    min_rel_w, max_rel_w = args.min_rel_w_full, args.max_rel_w_full
                else:
                    min_rel_w, max_rel_w = args.min_rel_w_win, args.max_rel_w_win

                neg = Image.open(neg_path)
                neg_t = transform_foreground(
                    fg=neg,
                    bg_w=bg_w,
                    bg_h=bg_h,
                    min_rel_w=min_rel_w,
                    max_rel_w=max_rel_w,
                    fg_jitter=args.fg_jitter,
                    twist_strength=args.twist_strength,
                    blur_prob=args.blur_prob,
                )

                bg, _ = paste_random(
                    bg_rgb=bg,
                    fg_rgba=neg_t,
                    allow_offscreen=args.allow_offscreen,
                    offscreen_frac=args.offscreen_frac,
                    quantize_xy=args.quantize_xy,
                )
            # labels remain empty

        bg = maybe_jpeg_artifacts(bg, args.jpeg_prob)

        stem = f"img_{i:06d}"
        img_out = out_root / "images" / split / f"{stem}.jpg"
        lab_out = out_root / "labels" / split / f"{stem}.txt"

        bg.save(img_out, quality=92)
        lab_out.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")

        if args.print_every > 0 and (i == 0 or (i + 1) % args.print_every == 0):
            print(f"[{i+1}/{args.n}] wrote {img_out.name} ({split})", flush=True)

    write_data_yaml(out_root)
    print("\nDone.", flush=True)
    print(f"Dataset: {out_root.resolve()}", flush=True)
    print(f"Train images: {args.n - val_n} | Val images: {val_n}", flush=True)
    print(f"data.yaml: {(out_root / 'data.yaml').resolve()}", flush=True)


if __name__ == "__main__":
    main()
