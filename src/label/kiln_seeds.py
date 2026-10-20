"""Seed kiln locations from SentinelKilnDB (Mondal et al., NeurIPS 2025 D&B; CC-BY-NC-4.0).

Reads only the label columns of the Hugging Face parquet files (no imagery), keeps kilns inside an AOI,
and converts patch-relative OBBs to lon/lat points + approximate footprints.

Patch convention (README): 128x128 px Sentinel-2 at 10 m, file name "<lat>_<lon>.png" — we assume the name is the
patch CENTRE; verify against high-res imagery (src.label.kiln_seeds --check) before trusting offsets.

    python -m src.label.kiln_seeds --aoi delhi_kiln_belt
"""
import argparse
import json
import math
import pathlib

import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfFileSystem

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPO = "datasets/SustainabilityLabIITGN/SentinelKilnDB"
PATCH_M, PATCH_PX = 1280.0, 128
CLASS_NAMES = {"0": "CFCBK", "1": "FCBK", "2": "Zigzag"}  # assumed README order — verify visually


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", default="delhi_kiln_belt")
    a = ap.parse_args()
    x0, y0, x1, y1 = yaml.safe_load(open(ROOT / "configs/aoi.yaml"))["aois"][a.aoi]["bbox"]

    fs = HfFileSystem()
    feats, seen = [], set()
    for split in ["train", "val", "test"]:
        f = pq.ParquetFile(fs.open(f"{REPO}/{split}/{split}.parquet"))
        for g in range(f.num_row_groups):
            for r in f.read_row_group(g, columns=["image_name", "yolo_obb_label"]).to_pylist():
                lat, lon = map(float, r["image_name"].removesuffix(".png").split("_"))
                if not (x0 - 0.02 <= lon <= x1 + 0.02 and y0 - 0.02 <= lat <= y1 + 0.02) or not r["yolo_obb_label"]:
                    continue
                for line in r["yolo_obb_label"]:
                    p = line.replace(",", " ").split()
                    cls, xy = p[0], list(map(float, p[1:9]))
                    if max(xy) > 1.5:  # pixel coords rather than normalised
                        xy = [v / PATCH_PX for v in xy]
                    ring = []
                    for i in range(0, 8, 2):
                        dx, dy = (xy[i] - 0.5) * PATCH_M, (0.5 - xy[i + 1]) * PATCH_M
                        ring.append([lon + dx / (111320 * math.cos(math.radians(lat))), lat + dy / 110540])
                    cx, cy = sum(p[0] for p in ring) / 4, sum(p[1] for p in ring) / 4
                    key = (round(cx, 4), round(cy, 4))  # patches overlap -> dedupe at ~10 m
                    if key in seen or not (x0 <= cx <= x1 and y0 <= cy <= y1):
                        continue
                    seen.add(key)
                    feats.append({"type": "Feature",
                                  "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]},
                                  "properties": {"class": "brick_kiln", "kiln_type": CLASS_NAMES.get(cls, cls), "kiln_cls_raw": cls, "split": split,
                                                 "source": "SentinelKilnDB (S2 Nov2023-Feb2024)"}})
        print(split, "->", len(feats), "kilns so far", flush=True)
    out = ROOT / "data/labels" / f"kiln_seeds_{a.aoi}.geojson"
    json.dump({"type": "FeatureCollection", "features": feats}, open(out, "w"))
    from collections import Counter
    print(len(feats), "kilns", Counter(f["properties"]["kiln_type"] for f in feats), "->", out)


if __name__ == "__main__":
    main()
