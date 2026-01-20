#!/usr/bin/env python3
"""
make_best_dataset.py

Creates a "best" dataset (per your brief) by running generate_dataset.py in two passes:
  1) Fullscreen-heavy (large overlays, minimal offscreen crop)
  2) Windowed-heavy (more small/medium overlays, more offscreen crop)

Then merges both into a single final dataset folder:
  dataset/
    images/train, images/val
    labels/train, labels/val
    data.yaml

This version is UPDATED for your latest generate_dataset.py (supports:
--p-positive, --p-negative-window, --min-rel-w-full/win, etc.)

Usage:
  python make_best_dataset.py

Optional overrides:
  python make_best_dataset.py --assets assets --generator generate_dataset.py --final dataset --n1 4000 --n2 4000
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_clean_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def merge_dataset(src: Path, dst: Path) -> None:
    """
    Merge YOLO dataset by copying images/labels.
    To avoid filename collisions (img_000000...), we prefix with the src folder name.
    """
    for split in ("train", "val"):
        src_img = src / "images" / split
        src_lbl = src / "labels" / split
        dst_img = dst / "images" / split
        dst_lbl = dst / "labels" / split

        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        for img_path in src_img.iterdir():
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in IMG_EXTS:
                continue

            stem = img_path.stem
            new_stem = f"{src.name}_{stem}"

            new_img = dst_img / f"{new_stem}{img_path.suffix.lower()}"
            shutil.copy2(img_path, new_img)

            lbl_path = src_lbl / f"{stem}.txt"
            new_lbl = dst_lbl / f"{new_stem}.txt"
            if lbl_path.exists():
                shutil.copy2(lbl_path, new_lbl)
            else:
                # Shouldn't happen, but keep dataset valid
                new_lbl.write_text("", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=str, default="assets")
    ap.add_argument("--generator", type=str, default="generate_dataset.py")

    ap.add_argument("--final", type=str, default="dataset")
    ap.add_argument("--run1", type=str, default="dataset_full")
    ap.add_argument("--run2", type=str, default="dataset_win")

    ap.add_argument("--n1", type=int, default=4000)
    ap.add_argument("--n2", type=int, default=4000)
    ap.add_argument("--val-ratio", type=float, default=0.2)

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assets = Path(args.assets)
    gen = Path(args.generator)
    out1 = Path(args.run1)
    out2 = Path(args.run2)
    final = Path(args.final)

    if not assets.exists():
        raise FileNotFoundError(f"Assets not found: {assets.resolve()}")
    if not gen.exists():
        raise FileNotFoundError(f"Generator not found: {gen.resolve()}")

    # Clean old outputs
    ensure_clean_dir(out1)
    ensure_clean_dir(out2)
    ensure_clean_dir(final)

    # -------------------------
    # Pass 1: Fullscreen-heavy
    # -------------------------
    run([
        "python", str(gen),
        "--assets", str(assets),
        "--out", str(out1),
        "--n", str(args.n1),
        "--val-ratio", str(args.val_ratio),
        "--seed", str(args.seed),

        "--p-positive", "0.60",
        "--p-negative-window", "0.90",
        "--max-overlays", "1",

        "--allow-offscreen",
        "--offscreen-frac", "0.05",  # small crop so fullscreen stays mostly visible
        "--quantize-xy", "2",

        "--min-rel-w-full", "0.85",
        "--max-rel-w-full", "1.10",
        "--min-rel-w-win",  "0.45",
        "--max-rel-w-win",  "0.85",

        "--bg-jitter", "0.30",
        "--fg-jitter", "0.40",
        "--twist-strength", "0.15",
        "--blur-prob", "0.08",
        "--jpeg-prob", "0.35",

        "--print-every", "200",
    ])

    # -----------------------
    # Pass 2: Windowed-heavy
    # -----------------------
    run([
        "python", str(gen),
        "--assets", str(assets),
        "--out", str(out2),
        "--n", str(args.n2),
        "--val-ratio", str(args.val_ratio),
        "--seed", str(args.seed + 1),

        "--p-positive", "0.60",
        "--p-negative-window", "0.90",
        "--max-overlays", "1",

        "--allow-offscreen",
        "--offscreen-frac", "0.12",
        "--quantize-xy", "2",

        "--min-rel-w-full", "0.75",
        "--max-rel-w-full", "0.95",
        "--min-rel-w-win",  "0.25",
        "--max-rel-w-win",  "0.70",

        "--bg-jitter", "0.30",
        "--fg-jitter", "0.40",
        "--twist-strength", "0.20",
        "--blur-prob", "0.10",
        "--jpeg-prob", "0.40",

        "--print-every", "200",
    ])

    # -----------------------
    # Merge into final dataset
    # -----------------------
    (final / "images" / "train").mkdir(parents=True, exist_ok=True)
    (final / "images" / "val").mkdir(parents=True, exist_ok=True)
    (final / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (final / "labels" / "val").mkdir(parents=True, exist_ok=True)

    merge_dataset(out1, final)
    merge_dataset(out2, final)

    # data.yaml is identical across runs; copy from pass 1
    shutil.copy2(out1 / "data.yaml", final / "data.yaml")

    # Summary
    train_count = len(list((final / "images" / "train").iterdir()))
    val_count = len(list((final / "images" / "val").iterdir()))

    print("\n✅ Final merged dataset ready:", final.resolve(), flush=True)
    print("Train images:", train_count, flush=True)
    print("Val images:", val_count, flush=True)
    print("data.yaml:", (final / "data.yaml").resolve(), flush=True)
    print("\nNext step (example training):", flush=True)
    print("  yolo detect train data=dataset/data.yaml model=yolov8n.pt imgsz=960 epochs=50 batch=8", flush=True)


if __name__ == "__main__":
    main()
