"""Search and order Planet imagery (PlanetScope / SkySat) for an AOI via the CSDA-provisioned key.

Usage:
    export PL_API_KEY=...            # from the CSDA / Planet onboarding email
    python -m src.data.planet_download --aoi delhi_ncr --item-type SkySatCollect --max-cloud 0.1 --search-only
    python -m src.data.planet_download --aoi delhi_ncr --item-type PSScene --order
"""
import argparse
import json
import os
import pathlib
import time

import requests
import yaml

DATA_API = "https://api.planet.com/data/v1"
ORDERS_API = "https://api.planet.com/compute/ops/orders/v2"
BUNDLES = {"PSScene": "analytic_sr_udm2", "SkySatCollect": "pansharpened_udm2"}
ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_dotenv():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and not line.lstrip().startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def session():
    load_dotenv()
    key = os.environ.get("PL_API_KEY")
    if not key:
        raise SystemExit("Set PL_API_KEY (CSDA Planet key) in emission_inventory/.env or your environment.")
    s = requests.Session()
    s.auth = (key, "")
    return s


def bbox_geom(b):
    x0, y0, x1, y1 = b
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


def search(s, item_type, geom, start, end, max_cloud):
    flt = {"type": "AndFilter", "config": [
        {"type": "GeometryFilter", "field_name": "geometry", "config": geom},
        {"type": "DateRangeFilter", "field_name": "acquired",
         "config": {"gte": f"{start}T00:00:00Z", "lte": f"{end}T23:59:59Z"}},
        {"type": "RangeFilter", "field_name": "cloud_cover", "config": {"lte": max_cloud}},
    ]}
    r = s.post(f"{DATA_API}/quick-search", json={"item_types": [item_type], "filter": flt})
    r.raise_for_status()
    feats, page = [], r.json()
    while True:
        feats += page["features"]
        nxt = page["_links"].get("_next")
        if not nxt:
            return feats
        page = s.get(nxt).json()


def order(s, name, item_type, ids, geom):
    body = {
        "name": name,
        "products": [{"item_ids": ids, "item_type": item_type, "product_bundle": BUNDLES[item_type]}],
        "tools": [{"clip": {"aoi": geom}}],
    }
    r = s.post(ORDERS_API, json=body)
    r.raise_for_status()
    return r.json()["id"]


def wait_and_download(s, order_id, out):
    while True:
        o = s.get(f"{ORDERS_API}/{order_id}").json()
        print("order", order_id, o["state"])
        if o["state"] in ("success", "partial"):
            break
        if o["state"] == "failed":
            raise SystemExit(json.dumps(o, indent=2))
        time.sleep(60)
    for res in o["_links"]["results"]:
        dst = out / res["name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        with s.get(res["location"], stream=True) as r, open(dst, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--item-type", default="PSScene", choices=list(BUNDLES))
    ap.add_argument("--max-cloud", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--search-only", action="store_true")
    ap.add_argument("--order", action="store_true")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--tag", default="", help="suffix for the output folder, e.g. archive")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs/aoi.yaml"))
    geom = bbox_geom(cfg["aois"][a.aoi]["bbox"])
    start, end = cfg["dates"].get(a.aoi, cfg["dates"]["default"])
    start, end = a.start or start, a.end or end
    s = session()

    feats = search(s, a.item_type, geom, start, end, a.max_cloud)
    feats.sort(key=lambda f: f["properties"].get("cloud_cover", 1))
    out = ROOT / "data/raw/planet" / a.aoi / (a.item_type + (f"_{a.tag}" if a.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"type": "FeatureCollection", "features": feats}, open(out / "search.geojson", "w"))
    print(f"{len(feats)} {a.item_type} scenes; footprints -> {out/'search.geojson'}")

    if a.order and not a.search_only:
        ids = [f["id"] for f in feats[: a.limit]]
        oid = order(s, f"emis_{a.aoi}_{a.item_type}", a.item_type, ids, geom)
        wait_and_download(s, oid, out)


if __name__ == "__main__":
    main()
