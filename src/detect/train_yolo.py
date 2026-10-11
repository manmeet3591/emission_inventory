"""Train an oriented-box YOLO detector on multi-sensor chips.

Expects data/tiles/<dataset>/{images,labels}/{train,val} in Ultralytics OBB format
and a dataset yaml written by src/data/tile.py.

    python -m src.detect.train_yolo --data data/tiles/delhi_v1/dataset.yaml --model yolo11m-obb.pt --imgsz 1024
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11m-obb.pt")  # DOTA-pretrained OBB weights
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)  # "mps" on Mac, "0" on CUDA
    ap.add_argument("--name", default="emis_obb")
    a = ap.parse_args()

    YOLO(a.model).train(
        data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device=a.device,
        project="outputs/runs", name=a.name,
        # overhead imagery: rotations/flips are label-preserving; avoid colour jitter that fakes smoke
        degrees=180, flipud=0.5, fliplr=0.5, hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, mosaic=1.0,
    )


if __name__ == "__main__":
    main()
