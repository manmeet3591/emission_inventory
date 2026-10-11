"""Turn object detections into a 300 m gridded emission inventory.

Input : detections GeoJSON (points or polygons, EPSG:4326) with properties
        {class, score, state?, area_m2?, count?, sensor}
        + configs/emission_factors.yaml  {class: {activity_per_unit: ..., ef: {species: kg/activity}}}
Output: outputs/<aoi>_emissions_300m.npz  (one 2-D array per species, kg/yr per cell) + grid metadata

    python -m src.inventory.grid300 --aoi delhi_ncr --detections outputs/delhi_ncr_detections.geojson
"""
import argparse
import json
import pathlib

import numpy as np
import yaml
from pyproj import Transformer

ROOT = pathlib.Path(__file__).resolve().parents[2]


def centroid(geom):
    if geom["type"] == "Point":
        return geom["coordinates"]
    ring = np.asarray(geom["coordinates"][0] if geom["type"] == "Polygon" else geom["coordinates"][0][0])
    return ring[:, 0].mean(), ring[:, 1].mean()


def activity(props, cls_cfg, ef_cfg):
    """Activity units for one detection. Units are defined per class in emission_factors.yaml."""
    kind = cls_cfg["activity"]
    base = ef_cfg.get("activity_per_unit", 1.0)
    if kind in ("per_unit", "facility"):
        a = base
    elif kind == "per_area":
        a = base * props.get("area_m2", 0.0)
    elif kind == "count_vehicles":
        a = base * props.get("count", 1)
    elif kind == "per_length":
        a = base * props.get("length_m", 0.0)
    else:
        raise ValueError(kind)
    a *= ef_cfg.get("state_multiplier", {}).get(props.get("state"), 1.0)
    return a * props.get("score", 1.0) if ef_cfg.get("score_weighted") else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--ef", default=str(ROOT / "configs/emission_factors.yaml"))
    a = ap.parse_args()

    aoi_cfg = yaml.safe_load(open(ROOT / "configs/aoi.yaml"))
    tax = yaml.safe_load(open(ROOT / "configs/taxonomy.yaml"))
    efs = yaml.safe_load(open(a.ef))
    res = aoi_cfg["grid_res_m"]
    aoi = aoi_cfg["aois"][a.aoi]
    to_grid = Transformer.from_crs("EPSG:4326", aoi["grid_crs"], always_xy=True)

    x0, y0, x1, y1 = aoi["bbox"]
    xs, ys = to_grid.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    gx0, gy1 = np.floor(min(xs) / res) * res, np.ceil(max(ys) / res) * res
    nx = int(np.ceil((max(xs) - gx0) / res))
    ny = int(np.ceil((gy1 - min(ys)) / res))
    grids = {sp: np.zeros((ny, nx), np.float64) for sp in tax["species"]}

    skipped = 0
    for f in json.load(open(a.detections))["features"]:
        p = f["properties"]
        c = p["class"]
        if c not in efs:
            skipped += 1
            continue
        lon, lat = centroid(f["geometry"])
        x, y = to_grid.transform(lon, lat)
        col, row = int((x - gx0) // res), int((gy1 - y) // res)
        if not (0 <= row < ny and 0 <= col < nx):
            continue
        act = activity(p, tax["classes"][c], efs[c])
        for sp, ef in efs[c]["ef"].items():
            grids[sp][row, col] += act * ef

    out = ROOT / "outputs" / f"{a.aoi}_emissions_300m.npz"
    np.savez_compressed(out, **grids, transform=np.array([gx0, res, 0, gy1, 0, -res]), crs=aoi["grid_crs"])
    print(f"wrote {out} ({ny}x{nx}); {skipped} detections had no EF entry")
    for sp, g in grids.items():
        if g.any():
            print(f"  {sp}: total {g.sum():.3e} kg/yr")


if __name__ == "__main__":
    main()
