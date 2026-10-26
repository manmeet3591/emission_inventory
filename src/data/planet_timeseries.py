"""PlanetScope downloads for (a) same-day pairs with SkySat scenes and (b) per-site time series.

  pairs : closest PSScene (±1 day) to each SkySat scene in a selection file, clipped to the SkySat footprints
  sites : small buffers around point sites (e.g. kilns) → per cluster, the N clearest scenes per month,
          clipped to the buffers (quota is charged on the clipped area)

Both order analytic_sr_udm2 (surface reflectance + usable-data mask) harmonized to Sentinel-2 so the
PS2 / PS2.SD / PSB.SD sensor generations are comparable.

    python -m src.data.planet_timeseries pairs --selection data/raw/planet/delhi_ncr/pilot_selection.json --name delhi_pilot_pairs
    python -m src.data.planet_timeseries sites --sites a.geojson b.geojson --name kilns_phase1 --start 2017-01-01 --per-month 2
"""
import argparse
import collections
import datetime as dt
import json
import pathlib

import requests
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Point, mapping, shape
from shapely.ops import transform, unary_union

from src.data.planet_download import DATA_API, ROOT, order, session, wait_and_download

TOOLS = [{"harmonize": {"target_sensor": "Sentinel-2"}}]


def has_sr(f):
    """True if the analytic_sr_udm2 bundle's assets are downloadable for this item (not all older scenes have them)."""
    p = set(f.get("_permissions", []))
    sr = {"assets.ortho_analytic_4b_sr:download", "assets.ortho_analytic_8b_sr:download"} & p
    return bool(sr) and "assets.ortho_udm2:download" in p


def search_all(s, geom, start, end, max_cloud=0.1):
    flt = {"type": "AndFilter", "config": [
        {"type": "GeometryFilter", "field_name": "geometry", "config": geom},
        {"type": "DateRangeFilter", "field_name": "acquired", "config": {"gte": start, "lte": end}},
        {"type": "RangeFilter", "field_name": "cloud_cover", "config": {"lte": max_cloud}},
        {"type": "PermissionFilter", "config": ["assets:download"]}]}
    r = s.post(f"{DATA_API}/quick-search?_page_size=250", json={"item_types": ["PSScene"], "filter": flt})
    r.raise_for_status()
    page, feats = r.json(), []
    while True:
        feats += page["features"]
        nxt = page["_links"].get("_next")
        if not nxt:
            return feats
        page = s.get(nxt).json()


def place_order(s, name, ids, aoi):
    """Orders hold ≤500 items; returns order ids."""
    body_tools = [{"clip": {"aoi": mapping(aoi)}}] + TOOLS
    oids = []
    for i in range(0, len(ids), 450):
        r = s.post("https://api.planet.com/compute/ops/orders/v2", json={
            "name": f"{name}_{i // 450}",
            "products": [{"item_ids": ids[i:i + 450], "item_type": "PSScene", "product_bundle": "analytic_sr_udm2"}],
            "tools": body_tools})
        if not r.ok:
            raise SystemExit(f"order failed: {r.status_code} {r.text[:600]}")
        oids.append(r.json()["id"])
    return oids


def cmd_pairs(a, s):
    sel = json.load(open(a.selection))
    arch = {f["id"]: f for f in json.load(open(ROOT / "data/raw/planet/delhi_ncr/SkySatCollect_archive/search.geojson"))["features"]}
    ids, footprints, manifest = [], [], []
    for x in sel:
        f = arch[x["id"]]
        t0 = dt.datetime.fromisoformat(f["properties"]["acquired"].replace("Z", "+00:00"))
        g = shape(f["geometry"])
        cands = search_all(s, mapping(g.centroid), (t0 - dt.timedelta(days=1)).isoformat(),
                           (t0 + dt.timedelta(days=1)).isoformat(), max_cloud=0.2)
        cands = [c for c in cands if shape(c["geometry"]).intersection(g).area / g.area > 0.3]
        if not cands:
            print("no pair for", x["id"]); continue
        best = min(cands, key=lambda c: abs(dt.datetime.fromisoformat(c["properties"]["acquired"].replace("Z", "+00:00")) - t0))
        gap_h = abs(dt.datetime.fromisoformat(best["properties"]["acquired"].replace("Z", "+00:00")) - t0).total_seconds() / 3600
        ids.append(best["id"]); footprints.append(g)
        manifest.append({"skysat": x["id"], "planetscope": best["id"], "gap_hours": round(gap_h, 2),
                         "instrument": best["properties"].get("instrument"), "why": x.get("why")})
        print(f"{x['id']} ↔ {best['id']} ({gap_h:.1f} h, {best['properties'].get('instrument')})")
    out = ROOT / "data/raw/planet/delhi_ncr" / a.name
    out.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(out / "pairs.json", "w"), indent=1)
    return a.name, sorted(set(ids)), unary_union(footprints), out


def cmd_sites(a, s):
    pts = [shape(f["geometry"]).centroid for p in a.sites for f in json.load(open(p))["features"]]
    to_m = Transformer.from_crs(4326, 3857, always_xy=True).transform
    to_ll = Transformer.from_crs(3857, 4326, always_xy=True).transform
    bufs = [transform(to_ll, transform(to_m, p).buffer(a.buffer_m, cap_style=3)) for p in pts]
    aoi = unary_union(bufs)
    clusters = list(aoi.geoms) if isinstance(aoi, MultiPolygon) else [aoi]
    # group nearby blobs into search regions (~0.25° cells) to keep the number of searches small
    regions = collections.defaultdict(list)
    for c in clusters:
        regions[(round(c.centroid.x * 4), round(c.centroid.y * 4))].append(c)
    ids, manifest = set(), []
    for key, blobs in regions.items():
        reg = unary_union(blobs)
        feats = search_all(s, mapping(reg.envelope), f"{a.start}T00:00:00Z", f"{a.end}T23:59:59Z")
        n_all = len(feats)
        feats = [f for f in feats if has_sr(f)]
        print(f"  {n_all - len(feats)} of {n_all} candidates lack surface-reflectance/udm2 access — skipped")
        by_month = collections.defaultdict(list)
        for f in feats:
            cov = shape(f["geometry"]).intersection(reg).area / reg.area
            if cov >= a.min_cover:
                by_month[f["properties"]["acquired"][:7]].append((f["properties"].get("clear_percent", 0), cov, f))
        for m, lst in by_month.items():
            for _, cov, f in sorted(lst, key=lambda t: (-t[0], -t[1]))[: a.per_month]:
                if f["id"] not in ids:
                    ids.add(f["id"])
                    manifest.append({"id": f["id"], "acquired": f["properties"]["acquired"], "region": list(key),
                                     "clear_percent": f["properties"].get("clear_percent"), "cover": round(cov, 2),
                                     "instrument": f["properties"].get("instrument")})
        print(f"region {key}: {len(blobs)} site blobs, {len(feats)} candidates, {len(by_month)} months")
    out = ROOT / "data/raw/planet/timeseries" / a.name
    out.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(out / "items.json", "w"), indent=1)
    json.dump({"type": "Feature", "geometry": mapping(aoi), "properties": {}}, open(out / "aoi.geojson", "w"))
    area = transform(to_m, aoi).area / 1e6 * 0.88 ** 2  # rough Web-Mercator → true area at ~28°N
    print(f"{len(ids)} scenes; clipped AOI ≈ {area:.1f} km² → quota ≈ {area * len(ids):,.0f} km² (upper bound)")
    return a.name, sorted(ids), aoi, out


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("pairs"); p.add_argument("--selection", required=True); p.add_argument("--name", required=True)
    q = sp.add_parser("sites"); q.add_argument("--sites", nargs="+", required=True); q.add_argument("--name", required=True)
    q.add_argument("--start", default="2017-01-01"); q.add_argument("--end", default=dt.date.today().isoformat())
    q.add_argument("--per-month", type=int, default=2); q.add_argument("--buffer-m", type=float, default=400)
    q.add_argument("--min-cover", type=float, default=0.8)
    for x in (p, q):
        x.add_argument("--dry-run", action="store_true"); x.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()
    s = session()
    name, ids, aoi, out = (cmd_pairs if a.cmd == "pairs" else cmd_sites)(a, s)
    if a.dry_run or not ids:
        print("dry run — nothing ordered"); return
    oids = place_order(s, name, ids, aoi)
    json.dump(oids, open(out / "order_ids.json", "w"))
    print("orders:", oids)
    if not a.no_wait:
        for oid in oids:
            wait_and_download(s, oid, out)


if __name__ == "__main__":
    main()
