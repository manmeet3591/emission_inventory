"""Per-kiln PlanetScope time series: reflectance of the kiln body and of the surrounding brick yard.

For every harmonized SR scene (B,G,R,NIR at 3 m) and every kiln it covers, records clear-pixel means inside the
kiln box ("body") and in a ring around it ("yard", where green bricks dry during the firing season).
Seasonal / event signals in these series are the activity proxy for the dynamic inventory.

    python -m src.activity.kiln_timeseries --scenes data/raw/planet/timeseries/kilns_phase1 \
        --kilns data/labels/kilns_pilot_scenes.geojson data/labels/kiln_seeds_lucknow_c3.geojson \
        --out outputs/activity/kiln_ts_phase1.csv
"""
import argparse
import csv
import glob
import json
import pathlib
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import mapping, shape
from shapely.ops import transform

RING_M = 150
FAR_M = (250, 380)  # control ring: same scene, same day, farmland around the kiln (other kilns excluded)


def load_kilns(paths):
    ks = []
    for p in paths:
        for i, f in enumerate(json.load(open(p))["features"]):
            ks.append({"kiln_id": f"{pathlib.Path(p).stem}:{i}", "geom": shape(f["geometry"]),
                       "kiln_type": f["properties"].get("kiln_type")})
    return ks


def scene(args):
    sr_path, kilns = args
    udm_path = sr_path.replace("AnalyticMS_SR_harmonized_clip", "udm2_clip")
    sid = pathlib.Path(sr_path).name.split("_3B_")[0]
    date = re.match(r"(\d{8})", sid).group(1)
    rows = []
    with rasterio.open(sr_path) as sr, rasterio.open(udm_path) as um:
        to = Transformer.from_crs(4326, sr.crs, always_xy=True).transform
        b = sr.bounds
        from shapely.ops import unary_union
        all_k = unary_union([transform(to, k["geom"]).buffer(RING_M) for k in kilns])
        for k in kilns:
            g = transform(to, k["geom"])
            ring = g.buffer(RING_M).difference(g)
            far = g.centroid.buffer(FAR_M[1]).difference(g.centroid.buffer(FAR_M[0])).difference(all_k)
            x0, y0, x1, y1 = far.bounds if not far.is_empty else ring.bounds
            if x0 < b.left or x1 > b.right or y0 < b.bottom or y1 > b.top:
                continue
            w = from_bounds(x0, y0, x1, y1, sr.transform).round_offsets().round_lengths()
            a = sr.read(window=w).astype(np.float32) / 10000.0
            clear = um.read(1, window=w) == 1
            if a.shape[1] < 5 or a.shape[2] < 5:
                continue
            T = sr.window_transform(w)
            valid = clear & (a.max(0) > 0)
            m_body = ~geometry_mask([mapping(g)], a.shape[1:], T) & valid
            m_yard = ~geometry_mask([mapping(ring)], a.shape[1:], T) & valid
            m_far = (~geometry_mask([mapping(far)], a.shape[1:], T) & valid) if not far.is_empty else np.zeros_like(valid)
            n_body = (~geometry_mask([mapping(g)], a.shape[1:], T)).sum()
            if n_body == 0 or m_body.sum() / n_body < 0.8 or m_yard.sum() < 50:
                continue
            row = {"kiln_id": k["kiln_id"], "kiln_type": k["kiln_type"], "date": date, "scene": sid,
                   "clear_body": round(float(m_body.sum() / n_body), 3)}
            if m_far.sum() < 100:
                continue
            for part, m in (("body", m_body), ("yard", m_yard), ("far", m_far)):
                bgrn = [float(a[i][m].mean()) for i in range(4)]
                row.update({f"{part}_{c}": round(v, 5) for c, v in zip("bgrn", bgrn)})
                row[f"{part}_ndvi"] = round((bgrn[3] - bgrn[2]) / (bgrn[3] + bgrn[2] + 1e-6), 4)
                row[f"{part}_bright"] = round(sum(bgrn[:3]) / 3, 5)
                row[f"{part}_red_green"] = round(bgrn[2] / (bgrn[1] + 1e-6), 4)
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--kilns", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    kilns = load_kilns(a.kilns)
    srs = sorted(glob.glob(f"{a.scenes}/**/*_AnalyticMS_SR_harmonized_clip.tif", recursive=True))
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n, w = 0, None
    with ProcessPoolExecutor(a.workers) as ex, open(out, "w", newline="") as fh:
        for i, rows in enumerate(ex.map(scene, [(p, kilns) for p in srs], chunksize=4)):
            for r in rows:
                if w is None:
                    w = csv.DictWriter(fh, fieldnames=list(r))
                    w.writeheader()
                w.writerow(r)
                n += 1
            if i % 100 == 0:
                print(f"{i}/{len(srs)} scenes, {n} kiln-observations", flush=True)
    print(f"done: {n} kiln-observations from {len(srs)} scenes, {len(kilns)} kilns -> {out}")


if __name__ == "__main__":
    main()
