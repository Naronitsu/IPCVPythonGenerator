#!/usr/bin/env python3
"""
run_yolo.py

Run the trained YOLOv8 chatbot detector on images, folders, or videos.
Uses the latest trained model automatically.

Outputs are saved to:
runs/detect/predict_chatbots/
"""

from pathlib import Path
import argparse
from ultralytics import YOLO


# Fixed model path (your trained model)
MODEL_PATH = Path("runs/detect/chatbots_yolov8/weights/best.pt")



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, required=True,
                    help="Image, folder, or video file")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--device", type=str, default="0")  # use 'cpu' if needed
    ap.add_argument("--name", type=str, default="predict_chatbots")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH.resolve()}\n"
            f"Did training finish successfully?"
        )

    model = YOLO(str(MODEL_PATH))

    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        project="runs",
        name=args.name,
        verbose=True,
    )

    out_dir = Path("") / args.name
    print(f"\Done. Outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
