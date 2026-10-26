"""Label Studio export -> Ultralytics YOLO-OBB dataset.

Gold chips (human-edited, data/labels/gold_chips.json) go to val — or, with --val-scenes, only the gold chips
of those scenes, so the rest of the human corrections are used for training. All other submitted chips → train.
Images are symlinked from data/tiles; empty label files mark verified negatives.

    python -m src.detect.build_dataset data/labels/ls_export_latest.json --name v0
"""
import argparse
import json
import pathlib

import yaml

from src.label.ls_bridge import OBJ_CLASSES, ls_to_poly

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ls_json")
    ap.add_argument("--name", required=True)
    ap.add_argument("--gold", default=str(ROOT / "data/labels/gold_chips.json"))
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--val-scenes", nargs="*", default=[],
                    help="only gold chips from these scene-id prefixes are held out (spatially disjoint test)")
    a = ap.parse_args()

    gold = set(json.load(open(a.gold)))
    out = ROOT / "data/yolo" / a.name
    counts = {s: {"chips": 0, "boxes": 0} for s in ("train", "val")}
    for t in json.load(open(a.ls_json)):
        anns = [x for x in t.get("annotations", []) if not x.get("was_cancelled")]
        if not anns:
            continue
        chip, tiles = t["data"]["chip"], t["data"]["tiles"]
        held_out = chip in gold and (not a.val_scenes or any(tiles.startswith(s) for s in a.val_scenes))
        split = "val" if held_out else "train"
        lines = []
        for r in anns[-1]["result"]:
            if r["type"] != "rectanglelabels":
                continue
            poly = ls_to_poly(r["value"], a.size, a.size).clip(0, a.size - 1)
            lines.append(f"{OBJ_CLASSES.index(r['value']['rectanglelabels'][0])} "
                         + " ".join(f"{v / a.size:.6f}" for v in poly.ravel()))
        for sub in ("images", "labels"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)
        img = out / "images" / split / chip
        if not img.exists():
            img.symlink_to(ROOT / "data/tiles" / tiles / "images" / chip)
        (out / "labels" / split / chip.replace(".jpg", ".txt")).write_text("\n".join(lines))
        counts[split]["chips"] += 1
        counts[split]["boxes"] += len(lines)
    yaml.safe_dump({"path": str(out), "train": "images/train", "val": "images/val",
                    "names": dict(enumerate(OBJ_CLASSES))}, open(out / "dataset.yaml", "w"), sort_keys=False)
    print(counts, "->", out / "dataset.yaml")


if __name__ == "__main__":
    main()
