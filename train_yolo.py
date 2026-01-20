from ultralytics import YOLO
from pathlib import Path

def main():
    data_yaml = Path("dataset/data.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing {data_yaml}. Did you generate the dataset?")

    # Choose base model: yolov8n.pt (fast) or yolov8s.pt (better)
    model = YOLO("yolov8n.pt")

    # Train with early stopping (patience)
    model.train(
        data=str(data_yaml),
        imgsz=960,          # good for small UI details; drop to 640 if slow/oom
        epochs=100,         # upper bound; early stop will cut it
        patience=20,        # <-- early stopping patience (epochs without improvement)
        batch=8,            # adjust if you get OOM (try 4)
        device=0,           # set to "cpu" if no GPU
        workers=4,
        pretrained=True,
        optimizer="SGD",    # stable default; try "AdamW" if you want
        lr0=0.01,           # initial LR
        cos_lr=True,        # cosine schedule (nice for stability)
        project="runs/detect",
        name="chatbots_yolov8",
        verbose=True
    )

    # Validate best checkpoint (writes metrics + plots)
    best = Path("runs/detect/chatbots_yolov8/weights/best.pt")
    if best.exists():
        YOLO(str(best)).val(data=str(data_yaml), imgsz=960, device=0)
    else:
        print("Best checkpoint not found (training may have failed).")

if __name__ == "__main__":
    main()
