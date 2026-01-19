import random
import math
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

# --- config ---
BG_DIR = Path("assets/bg")
FG_ROOT = Path("assets/fg")  # subfolders: chatgpt, claude, copilot
OUT_ROOT = Path("dataset")
IMG_SIZE_LIMIT = 1920  # downscale huge backgrounds for consistency

CLASSES = ["chatgpt", "claude", "copilot"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

N_IMAGES = 4000
VAL_RATIO = 0.2

# probability controls
P_NEGATIVE = 0.15          # images with no overlays (background-only)
MAX_OVERLAYS = 3
P_OCCLUDE = 0.25

random.seed(42)

def load_paths(dir_path):
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return [p for p in dir_path.rglob("*") if p.suffix.lower() in exts]

BG_PATHS = load_paths(BG_DIR)
FG_PATHS = {c: load_paths(FG_ROOT / c) for c in CLASSES}

if not BG_PATHS:
    raise SystemExit("No backgrounds found in assets/bg")
for c in CLASSES:
    if not FG_PATHS[c]:
        raise SystemExit(f"No foregrounds found in assets/fg/{c}")

def maybe_downscale(im: Image.Image) -> Image.Image:
    w, h = im.size
    m = max(w, h)
    if m <= IMG_SIZE_LIMIT:
        return im
    scale = IMG_SIZE_LIMIT / m
    return im.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)

def rand_transform(fg: Image.Image, bg_w: int, bg_h: int):
    # scale relative to background
    # keep it between ~3% and ~20% of bg width (adjust if you want)
    target_w = random.uniform(0.03, 0.20) * bg_w
    scale = target_w / max(1, fg.size[0])
    new_w = max(8, int(fg.size[0] * scale))
    new_h = max(8, int(fg.size[1] * scale))
    fg2 = fg.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # slight rotation
    angle = random.uniform(-5, 5)
    fg2 = fg2.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

    # opacity jitter (ensure RGBA)
    if fg2.mode != "RGBA":
        fg2 = fg2.convert("RGBA")
    alpha = fg2.split()[-1]
    alpha = ImageEnhance.Brightness(alpha).enhance(random.uniform(0.7, 1.0))
    fg2.putalpha(alpha)

    # optional blur
    if random.random() < 0.2:
        fg2 = fg2.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))

    # random position (keep fully inside)
    fw, fh = fg2.size
    if fw >= bg_w or fh >= bg_h:
        # if too big, shrink more
        shrink = min((bg_w-2)/fw, (bg_h-2)/fh, 0.9)
        fg2 = fg2.resize((max(8, int(fw*shrink)), max(8, int(fh*shrink))), Image.Resampling.LANCZOS)
        fw, fh = fg2.size

    x0 = random.randint(0, bg_w - fw)
    y0 = random.randint(0, bg_h - fh)

    return fg2, x0, y0

def apply_jpeg_artifacts(im: Image.Image) -> Image.Image:
    # simulate compression sometimes by round-trip saving to memory
    if random.random() < 0.35:
        from io import BytesIO
        buf = BytesIO()
        q = random.randint(35, 85)
        im.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    return im

def occlude(bg: Image.Image, bbox):
    # bbox: (x1,y1,x2,y2)
    if random.random() >= P_OCCLUDE:
        return bg
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    # occlude 10–40% of the overlay region with a rectangle
    occ_w = int(w * random.uniform(0.1, 0.4))
    occ_h = int(h * random.uniform(0.1, 0.4))
    ox = random.randint(x1, max(x1, x2 - occ_w))
    oy = random.randint(y1, max(y1, y2 - occ_h))
    # draw a solid rectangle with random gray-ish color
    rect = Image.new("RGB", (occ_w, occ_h), (random.randint(0,255),)*3)
    bg.paste(rect, (ox, oy))
    return bg

def yolo_line(class_id, bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    xc = (x1 + x2) / 2.0 / img_w
    yc = (y1 + y2) / 2.0 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    # clamp (safety)
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)
    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

def ensure_dirs(split):
    (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

ensure_dirs("train")
ensure_dirs("val")

def generate_one(i, split):
    bg_path = random.choice(BG_PATHS)
    bg = Image.open(bg_path).convert("RGB")
    bg = maybe_downscale(bg)
    bg_w, bg_h = bg.size

    labels = []

    if random.random() > P_NEGATIVE:
        n = random.randint(1, MAX_OVERLAYS)
        for _ in range(n):
            cls = random.choice(CLASSES)
            fg_path = random.choice(FG_PATHS[cls])
            fg = Image.open(fg_path)
            if fg.mode != "RGBA":
                fg = fg.convert("RGBA")

            fg2, x0, y0 = rand_transform(fg, bg_w, bg_h)

            # paste using alpha
            bg_rgba = bg.convert("RGBA")
            bg_rgba.alpha_composite(fg2, (x0, y0))
            bg = bg_rgba.convert("RGB")

            # bbox after rotation/resize is just fg2 extents
            fw, fh = fg2.size
            bbox = (x0, y0, x0 + fw, y0 + fh)

            # occasionally occlude after paste
            bg = occlude(bg, bbox)

            labels.append((CLASS_TO_ID[cls], bbox))

    # final artifacts
    bg = apply_jpeg_artifacts(bg)

    stem = f"img_{i:06d}"
    img_out = OUT_ROOT / "images" / split / f"{stem}.jpg"
    lab_out = OUT_ROOT / "labels" / split / f"{stem}.txt"

    bg.save(img_out, quality=92)

    with open(lab_out, "w", encoding="utf-8") as f:
        for class_id, bbox in labels:
            f.write(yolo_line(class_id, bbox, bg_w, bg_h) + "\n")

def main():
    indices = list(range(N_IMAGES))
    random.shuffle(indices)
    val_n = int(N_IMAGES * VAL_RATIO)
    val_set = set(indices[:val_n])

    for i in range(N_IMAGES):
        split = "val" if i in val_set else "train"
        generate_one(i, split)

    # write data.yaml
    yaml = OUT_ROOT / "data.yaml"
    yaml.write_text(
        "path: dataset\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: chatgpt\n"
        "  1: claude\n"
        "  2: copilot\n",
        encoding="utf-8"
    )
    print(f"Done. Wrote {N_IMAGES} images to {OUT_ROOT}/ and data.yaml")

if __name__ == "__main__":
    main()
