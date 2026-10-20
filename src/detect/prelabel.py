"""Generate candidate boxes on chips to speed up fresh labelling.

Two zero-/few-shot sources, both reviewed by a human before becoming labels:
  * DOTA-pretrained YOLO-OBB  -> vehicles, planes, storage tanks, ships (directly useful classes)
  * YOLO-World open-vocabulary -> text prompts for emitter categories (noisy on overhead imagery; recall aid only)

    python -m src.detect.prelabel data/tiles/lucknow_20260115_rgb --device mps

Writes <tiles>/prelabels.json: {chip: [{cls, conf, source, obb|xyxy}, ...]}
"""
import argparse
import json
import pathlib

from ultralytics import YOLO, YOLOWorld

WEIGHTS = pathlib.Path(__file__).resolve().parents[2] / "weights"

# DOTA class name -> our taxonomy (None = ignore)
DOTA_MAP = {
    "plane": "aircraft_parked", "ship": "ship_anchored", "storage tank": "storage_tank",
    "large vehicle": "large_vehicle", "small vehicle": "small_vehicle",
}
WORLD_PROMPTS = {
    "brick kiln": "brick_kiln", "industrial chimney": "chimney", "smoke plume": "smoke_plume",
    "garbage dump": "landfill", "construction site": "construction_site", "cremation ground": "crematorium",
    "truck parking lot": "truck_parking_hub", "bus depot": "bus_depot", "cattle shed": "dairy_colony",
    "stone crusher": "stone_crusher", "sand pile": "aggregate_stockpile", "burnt field": "crop_residue_burn_scar",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tiles")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--conf", type=float, default=0.15)
    a = ap.parse_args()

    tiles = pathlib.Path(a.tiles)
    chips = sorted((tiles / "images").glob("*.jpg"))
    obb = YOLO(WEIGHTS / "yolo11m-obb.pt")
    world = YOLOWorld(WEIGHTS / "yolov8x-worldv2.pt")
    world.set_classes(list(WORLD_PROMPTS))

    out = {}
    for i in range(0, len(chips), 16):
        batch = [str(p) for p in chips[i:i + 16]]
        for p, r in zip(batch, obb.predict(batch, imgsz=1024, conf=a.conf, device=a.device, verbose=False)):
            names = r.names
            for cls, conf, poly in zip(r.obb.cls.tolist(), r.obb.conf.tolist(), r.obb.xyxyxyxy.tolist()):
                tgt = DOTA_MAP.get(names[int(cls)])
                if tgt:
                    out.setdefault(pathlib.Path(p).name, []).append(
                        {"cls": tgt, "conf": round(conf, 3), "source": "dota_obb", "obb": poly})
        for p, r in zip(batch, world.predict(batch, imgsz=1024, conf=a.conf, device=a.device, verbose=False)):
            for cls, conf, box in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                out.setdefault(pathlib.Path(p).name, []).append(
                    {"cls": WORLD_PROMPTS[list(WORLD_PROMPTS)[int(cls)]], "conf": round(conf, 3),
                     "source": "yolo_world", "xyxy": box})
        print(f"{min(i + 16, len(chips))}/{len(chips)}", end="\r")
    json.dump(out, open(tiles / "prelabels.json", "w"))
    n = sum(map(len, out.values()))
    print(f"\n{n} candidate boxes on {len(out)}/{len(chips)} chips -> {tiles/'prelabels.json'}")


if __name__ == "__main__":
    main()
