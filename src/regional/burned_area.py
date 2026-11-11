"""Crop-residue burned area from HLS (30 m) in the Punjab test boxes, to compare against VIIRS fire counts.

VIIRS fire counts upwind of Delhi fell ~95% from 2021 to 2025. VIIRS passes ~13:30 local, so burning moved to
late afternoon would vanish from the counts but not from the ground: a burnt field stays dark (char) for days
until it is ploughed. This module caches every clear HLS observation of each box (Sept–mid-Dec) on a fixed
UTM 30 m grid, so burn scars can be counted per season independent of the satellite fire detections.

    python -m src.regional.burned_area fetch --years 2019-2025
    python -m src.regional.burned_area analyse
"""
import argparse
import json
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from src.activity.hls_hotspots import GDAL_ENV, search, signed
from src.data.planet_download import load_dotenv

BOXES = pathlib.Path("data/derived/regional/burn_boxes.json")
CACHE = pathlib.Path("data/derived/regional/burn_hls")
CRS = "EPSG:32643"  # UTM 43N covers all four boxes
RES = 30
BANDS = {"HLSS30_2.0": {"nir": "B8A", "swir1": "B11", "swir2": "B12"},
         "HLSL30_2.0": {"nir": "B05", "swir1": "B06", "swir2": "B07"}}


def box_grid(bbox):
    x0, y0, x1, y1 = transform_bounds(4326, CRS, *bbox)
    x0, y1 = np.floor(x0 / RES) * RES, np.ceil(y1 / RES) * RES
    w, h = int(np.ceil((x1 - x0) / RES)), int(np.ceil((y1 - y0) / RES))
    return from_origin(x0, y1, RES, RES), w, h


def read_grid(session, href, transform, w, h):
    with rasterio.Env(**GDAL_ENV), rasterio.open(signed(session, href)) as d, \
            WarpedVRT(d, crs=CRS, transform=transform, width=w, height=h, resampling=Resampling.nearest,
                      nodata=d.nodata if d.nodata is not None else 255) as v:
        return v.read(1)


def fetch_one(args):
    item, box, token = args
    out = CACHE / box["name"] / f"{item['properties']['datetime'][:10]}_{item['collection'][:6]}_{item['id'][-30:]}.npz"
    if out.exists():
        return "cached"
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    T, w, h = box_grid(box["bbox"])
    try:
        fm = read_grid(s, item["assets"]["Fmask"]["href"], T, w, h).astype(np.uint8)
        fmi = fm.astype(np.int32)
        clear = (fm != 255) & ((fmi >> 1) & 0b111 == 0)
        if clear.mean() < 0.2:
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, skipped=True, clear_frac=clear.mean())
            return "cloudy"
        bands = {k: read_grid(s, item["assets"][v]["href"], T, w, h).astype(np.int16) for k, v in BANDS[item["collection"]].items()}
    except Exception as e:  # noqa: BLE001 — skip unreadable granules, report them
        return f"error {str(e)[:100]}"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, fmask=fm, **bands)
    return "ok"


def fetch(years, workers):
    load_dotenv()
    token = os.environ["EARTHDATA_TOKEN"]
    jobs = []
    for box in json.loads(BOXES.read_text()):
        for y in years:
            items = search(box["bbox"], f"{y}-09-01", f"{y}-12-15", max_cloud=101)
            print(box["name"], y, len(items), "granules", flush=True)
            jobs += [(i, box, token) for i in items]
    stats = {}
    with ThreadPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(fetch_one, jobs), 1):
            key = r.split()[0]
            stats[key] = stats.get(key, 0) + 1
            if key == "error":
                print(r, flush=True)
            if k % 100 == 0:
                print(k, "/", len(jobs), stats, flush=True)
    print("done", stats)


def load_box(name, year):
    """Clear-sky stack for one box and season: dates, NIR/SWIR1/SWIR2 reflectance (NaN where not clear)."""
    obs = {}
    for f in sorted((CACHE / name).glob(f"{year}-*.npz")):
        d = np.load(f)
        if "skipped" in d:
            continue
        fm = d["fmask"].astype(np.int32)
        clear = (fm != 255) & ((fm >> 1) & 0b111 == 0) & (d["nir"] > 0)
        bands = np.stack([np.where(clear, d[b] * 1e-4, np.nan) for b in ("nir", "swir1", "swir2")])
        date = np.datetime64(f.name[:10])
        obs[date] = bands if date not in obs else np.where(np.isnan(obs[date]), bands, obs[date])
    dates = np.array(sorted(obs))
    return dates, (np.stack([obs[t] for t in dates]) if len(dates) else None)


def burn_events(dates, X, nir_max=0.12, s2_max=0.16, drop_min=0.05, s1_min=0.06, gap_days=25):
    """Per observation, pixels that turned to char: dark in NIR and SWIR2 and darker than the previous clear look.

    Defaults calibrated on 621 VIIRS 2020 fires (box_75.4E_30.0N) seen clear 1–6 days later: char NIR median 0.09
    (95th pct 0.13), SWIR2 0.11; same-date random pixels NIR 0.21. SWIR1 > 0.06 excludes water.
    """
    nir, s1, s2 = X[:, 0], X[:, 1], X[:, 2]
    last_nir = np.full(nir.shape[1:], np.nan)
    last_t = np.full(nir.shape[1:], np.datetime64("NaT"), dtype="datetime64[D]")
    ev = np.zeros(nir.shape, bool)
    for k, t in enumerate(dates):
        ok = ~np.isnan(nir[k])
        recent = (t - last_t) <= np.timedelta64(gap_days, "D")
        ev[k] = ok & recent & (nir[k] < nir_max) & (s2[k] < s2_max) & (s1[k] > s1_min) & (last_nir - nir[k] > drop_min)
        last_nir = np.where(ok, nir[k], last_nir)
        last_t = np.where(ok, t, last_t)
    return ev


def max_gap_days(dates, X, start, end):
    """Per pixel, the longest run of days without a clear look inside [start, end] (a scar can hide in such a gap)."""
    sel = (dates >= start) & (dates <= end)
    ok = ~np.isnan(X[sel, 0])
    t = dates[sel].astype("datetime64[D]").astype(np.int64)
    edges = np.concatenate([[start.astype("datetime64[D]").astype(np.int64)], t, [end.astype("datetime64[D]").astype(np.int64)]])
    gap = np.zeros(ok.shape[1:], np.int64)
    last = np.full(ok.shape[1:], edges[0])
    for k in range(len(t)):
        gap = np.where(ok[k], np.maximum(gap, t[k] - last), gap)
        last = np.where(ok[k], t[k], last)
    return np.maximum(gap, edges[-1] - last)


def analyse(out_csv):
    import pandas as pd
    from pyproj import Transformer
    fires = pd.read_parquet("data/derived/regional/fires_region.parquet")
    fires = fires[(fires.type == 0) & (fires.confidence != "l")]
    rows = []
    for box in json.loads(BOXES.read_text()):
        x0, y0, x1, y1 = box["bbox"]
        fb = fires[fires.longitude.between(x0, x1) & fires.latitude.between(y0, y1)]
        for y in sorted({int(f.name[:4]) for f in (CACHE / box["name"]).glob("*.npz")}):
            dates, X = load_box(box["name"], y)
            sept = dates < np.datetime64(f"{y}-10-01")
            pre = np.nanmedian(np.where(sept[:, None, None], X[:, 0], np.nan), axis=0) if sept.any() else None
            nbr = (X[:, 0] - X[:, 2]) / (X[:, 0] + X[:, 2])
            crop = np.nanmedian(np.where(sept[:, None, None], nbr, np.nan), axis=0) > 0.3  # green in Sept = kharif crop
            gap = max_gap_days(dates, X, np.datetime64(f"{y}-10-15"), np.datetime64(f"{y}-11-30"))
            season = (dates >= np.datetime64(f"{y}-10-01"))
            fs = fb[(fb.t.dt.year == y) & (fb.t >= f"{y}-10-01") & (fb.t <= f"{y}-12-15")]
            row = dict(box=box["name"], year=y, clear_obs=int(season.sum()),
                       crop_km2=round(crop.sum() * 9e-4, 1),
                       median_max_gap_d=float(np.median(gap[crop])),
                       frac_crop_gap_le10d=round(float((gap[crop] <= 10).mean()), 3),
                       viirs_day=int((fs.daynight == "D").sum()), viirs_night=int((fs.daynight == "N").sum()))
            for tag, kw in {"": {}, "_strict": dict(nir_max=0.10, s2_max=0.14), "_loose": dict(nir_max=0.14, s2_max=0.18)}.items():
                ev = burn_events(dates, X, **kw)
                burned = ev[season].any(0) & crop
                row[f"burned_km2{tag}"] = round(burned.sum() * 9e-4, 1)
                row[f"sept_falsepos_km2{tag}"] = round((ev[sept].any(0) & crop).sum() * 9e-4, 2)
            row["burned_frac_crop"] = round(row["burned_km2"] / max(row["crop_km2"], 1e-6), 3)
            row["km2_per_day_fire"] = round(row["burned_km2"] / max(row["viirs_day"], 1), 3)
            print(row, flush=True)
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--years", default="2019-2025")
    f.add_argument("--workers", type=int, default=12)
    g = sub.add_parser("analyse")
    g.add_argument("--out", default="outputs/regional/burned_area_vs_viirs.csv")
    a = p.parse_args()
    if a.cmd == "fetch":
        y0, y1 = map(int, a.years.split("-"))
        fetch(range(y0, y1 + 1), a.workers)
    else:
        analyse(a.out)


if __name__ == "__main__":
    main()
