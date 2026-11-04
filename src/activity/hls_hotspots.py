"""Kiln heat signal from HLS (Harmonized Landsat–Sentinel-2, 30 m, NASA LP DAAC; free with an Earthdata token).

A firing kiln has a ~1000 °C combustion zone: sub-pixel at 30 m, but it lifts shortwave-infrared (SWIR2 more
than SWIR1) and, for Landsat, thermal brightness temperature. For every clear HLS granule and every kiln this
records the kiln's hottest 3×3 pixels vs the median of a 250–400 m ring of surrounding land on the same day.

Reads only small windows: LP DAAC URLs redirect to signed links, resolved here with the bearer token.

    python -m src.activity.hls_hotspots --kilns data/labels/kilns_pilot_scenes.geojson \
        --start 2019-10-01 --end 2021-09-30 --out outputs/activity/hls_kilns_2019_2021.csv
"""
import argparse
import collections
import csv
import json
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from shapely.geometry import shape

from src.data.planet_download import load_dotenv

STAC = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD/search"
BANDS = {"HLSS30_2.0": {"nir": "B8A", "swir1": "B11", "swir2": "B12"},
         "HLSL30_2.0": {"nir": "B05", "swir1": "B06", "swir2": "B07", "tir": "B10"}}
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS="tif",
                GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2")


def search(bbox, start, end, max_cloud):
    body = {"collections": list(BANDS), "bbox": bbox, "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 250}
    page, items = requests.post(STAC, json=body, timeout=120).json(), []
    while True:
        items += page["features"]
        nxt = [l for l in page.get("links", []) if l["rel"] == "next"]
        if not nxt:
            break
        n = nxt[0]
        page = (requests.post(n["href"], json=n.get("body", body), timeout=120) if n.get("method") == "POST"
                else requests.get(n["href"], timeout=120)).json()
    return [i for i in items if i["properties"].get("eo:cloud_cover", 100) <= max_cloud]


def signed(session, href):
    r = session.get(href, allow_redirects=False, timeout=120, stream=True)
    r.close()
    return r.headers.get("Location", href)


def read_window(session, href, bbox_ll):
    with rasterio.Env(**GDAL_ENV), rasterio.open(signed(session, href)) as d:
        b = transform_bounds(4326, d.crs, *bbox_ll)
        w = from_bounds(*b, d.transform).round_offsets().round_lengths()
        fill = d.nodata if d.nodata is not None else 0  # Fmask nodata 255 → treated as not clear
        a = d.read(1, window=w, boundless=True, fill_value=fill).astype(np.float32)
        if d.nodata is not None and d.nodata != -9999:
            a[a == d.nodata] = -9999 if "Fmask" not in href else 255
        return a, d.window_transform(w), d.crs


def granule(args):
    item, region_bbox, kilns, token = args
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    coll = item["collection"]
    try:
        fm, T, crs = read_window(s, item["assets"]["Fmask"]["href"], region_bbox)
        bands = {k: read_window(s, item["assets"][v]["href"], region_bbox)[0] for k, v in BANDS[coll].items()}
    except Exception as e:  # noqa: BLE001 — skip unreadable granules, report count
        return [], str(e)[:120]
    fmi = fm.astype(np.int32)
    clear = (fmi >= 0) & ((fmi >> 1) & 0b111 == 0)  # bits 1–3: cloud, adjacent cloud, cloud shadow
    for k in ("nir", "swir1", "swir2"):
        bands[k] = np.where(bands[k] == -9999, np.nan, bands[k] * 1e-4)
    if "tir" in bands:
        bands["tir"] = np.where(bands["tir"] == -9999, np.nan, bands["tir"] * 0.01)  # °C
    nhi = (bands["swir2"] - bands["swir1"]) / (bands["swir2"] + bands["swir1"] + 1e-6)
    inv = ~T
    to = Transformer.from_crs(4326, crs, always_xy=True).transform
    H, W = fm.shape
    yy, xx = np.mgrid[0:H, 0:W]
    px = np.array([inv * to(*k["xy"]) for k in kilns])  # (col, row)
    near_kiln = np.zeros((H, W), bool)
    for c, r in px:  # exclude all kilns (±150 m) from control rings
        near_kiln |= (xx - c) ** 2 + (yy - r) ** 2 <= 5 ** 2
    rows, date = [], item["properties"]["datetime"][:10]
    for k, (c, r) in zip(kilns, px):
        ci, ri = int(c), int(r)
        if not (8 <= ci < W - 8 and 8 <= ri < H - 8):
            continue
        sl = (slice(ri - 1, ri + 2), slice(ci - 1, ci + 2))
        if not clear[sl].all():
            continue
        d2 = (xx - c) ** 2 + (yy - r) ** 2
        ring = (d2 >= (250 / 30) ** 2) & (d2 <= (400 / 30) ** 2) & clear & ~near_kiln
        if ring.sum() < 30:
            continue
        row = {"kiln_id": k["kiln_id"], "date": date, "sensor": coll[:7], "granule": item["id"],
               "swir2_k": np.nanmax(bands["swir2"][sl]), "swir2_c": np.nanmedian(bands["swir2"][ring]),
               "swir1_k": np.nanmax(bands["swir1"][sl]), "swir1_c": np.nanmedian(bands["swir1"][ring]),
               "nhi_k": np.nanmax(nhi[sl]), "nhi_c": np.nanmedian(nhi[ring]),
               "nhi_c_p95": np.nanpercentile(nhi[ring], 95)}
        if "tir" in bands:
            row.update({"tir_k": np.nanmax(bands["tir"][sl]), "tir_c": np.nanmedian(bands["tir"][ring]),
                        "tir_c_p95": np.nanpercentile(bands["tir"][ring], 95)})
        rows.append({kk: (round(float(v), 5) if isinstance(v, (float, np.floating)) else v) for kk, v in row.items()})
    return rows, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kilns", nargs="+", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-cloud", type=float, default=40)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    load_dotenv()
    token = os.environ["EARTHDATA_TOKEN"]
    kilns = []
    for p in a.kilns:
        for i, f in enumerate(json.load(open(p))["features"]):
            c = shape(f["geometry"]).centroid
            kilns.append({"kiln_id": f"{pathlib.Path(p).stem}:{i}", "xy": (c.x, c.y)})
    regions = collections.defaultdict(list)  # ~0.25° cells keep windows small
    for k in kilns:
        regions[(round(k["xy"][0] * 4), round(k["xy"][1] * 4))].append(k)
    jobs = []
    for key, ks in regions.items():
        xs, ys = [k["xy"][0] for k in ks], [k["xy"][1] for k in ks]
        bbox = [min(xs) - 0.01, min(ys) - 0.01, max(xs) + 0.01, max(ys) + 0.01]
        items = search(bbox, a.start, a.end, a.max_cloud)
        print(f"region {key}: {len(ks)} kilns, {len(items)} granules (cloud ≤{a.max_cloud}%)", flush=True)
        jobs += [(it, bbox, ks, token) for it in items]
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n, errs, w = 0, collections.Counter(), None
    with ThreadPoolExecutor(a.workers) as ex, open(out, "w", newline="") as fh:
        for i, (rows, err) in enumerate(ex.map(granule, jobs)):
            if err:
                errs[err[:60]] += 1
            for r in rows:
                if w is None:
                    w = csv.DictWriter(fh, fieldnames=sorted({*r} | {"tir_k", "tir_c", "tir_c_p95"}), extrasaction="ignore")
                    w.writeheader()
                w.writerow(r)
                n += 1
            if i % 50 == 0:
                print(f"{i}/{len(jobs)} granules, {n} kiln-observations, errors {sum(errs.values())}", flush=True)
    print(f"done: {n} kiln-observations from {len(jobs)} granules -> {out}; errors: {dict(errs)}")


if __name__ == "__main__":
    main()
