"""Run a trained YOLO-OBB model over a tiled scene and write detections in map coordinates.

Overlapping chips produce duplicates; detections of the same class whose centres are closer than
--merge-m are merged (highest confidence kept).

    python -m src.detect.infer data/tiles/<stem> --weights runs/.../best.pt --out outputs/det/<stem>.geojson
"""
import argparse
import json
import pathlib

import numpy as np
from pyproj import Transformer
from rasterio.transform import Affine
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tiles")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--merge-m", type=float, default=30)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--imgsz", type=int, default=1024)
    a = ap.parse_args()

    tiles = pathlib.Path(a.tiles)
    idx = json.load(open(tiles / "index.json"))
    to_ll = Transformer.from_crs(idx["crs"], "EPSG:4326", always_xy=True)
    model = YOLO(a.weights)
    chips = sorted(idx["chips"])
    dets = []  # (class, conf, centre_xy_map, ring_map)
    for i in range(0, len(chips), 16):
        batch = chips[i:i + 16]
        for chip, r in zip(batch, model.predict([str(tiles / "images" / c) for c in batch], imgsz=a.imgsz,
                                                conf=a.conf, device=a.device, verbose=False)):
            T = Affine(*idx["chips"][chip]["transform"])
            for cls, conf, poly in zip(r.obb.cls.tolist(), r.obb.conf.tolist(), r.obb.xyxyxyxy.tolist()):
                ring = [T * (float(x), float(y)) for x, y in poly]
                dets.append((r.names[int(cls)], conf, np.mean(ring, axis=0), ring))
    dets.sort(key=lambda d: -d[1])
    kept = []
    for d in dets:
        if all(k[0] != d[0] or np.hypot(*(k[2] - d[2])) > a.merge_m for k in kept):
            kept.append(d)
    feats = []
    for cls, conf, c, ring in kept:
        ll = [list(to_ll.transform(x, y)) for x, y in ring]
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ll + [ll[0]]]},
                      "properties": {"class": cls, "score": round(conf, 3), "source": pathlib.Path(a.weights).parts[-3]}})
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"type": "FeatureCollection", "features": feats}, open(out, "w"))
    from collections import Counter
    print(f"{len(dets)} raw → {len(kept)} merged detections {dict(Counter(k[0] for k in kept))} → {out}")


if __name__ == "__main__":
    main()
