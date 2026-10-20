"""Bridge between our chips and Label Studio.

  tasks   : chips (+ prelabels.json) -> Label Studio import JSON, with candidates as editable predictions
  export  : Label Studio JSON export -> YOLO-OBB labels (objects) + GeoJSON in map coords (objects & areas)

    python -m src.label.ls_bridge tasks data/tiles/<stem> --base-url http://localhost:8081 --n 300
    python -m src.label.ls_bridge export ls_export.json --tiles data/tiles/<stem> --out data/labels/<stem>

Chip images must be served at <base-url>/<stem>/images/<chip>.jpg (python -m http.server in data/tiles).
"""
import argparse
import json
import math
import pathlib
import random

import numpy as np
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[2]
CFG = ET.parse(ROOT / "configs/labelstudio.xml").getroot()
OBJ_CLASSES = [l.get("value") for l in CFG.find("RectangleLabels")]
AREA_CLASSES = [l.get("value") for l in CFG.find("PolygonLabels")]
# prelabel class -> labelling class (None = drop)
PRE_MAP = {"brick_kiln": "brick_kiln", "chimney": "chimney_stack", "smoke_plume": "smoke_plume",
           "storage_tank": "storage_tank", "large_vehicle": "truck", "small_vehicle": "car",
           "aircraft_parked": "aircraft", "stone_crusher": "stone_crusher", "dairy_colony": "cattle_shed",
           "landfill": "landfill", "construction_site": "construction_site", "crematorium": "crematorium_site",
           "truck_parking_hub": "truck_parking_hub", "bus_depot": "bus_depot",
           "aggregate_stockpile": "aggregate_stockpile", "crop_residue_burn_scar": "crop_burn_scar"}


def poly_to_ls(pts, W, H):
    """4 corner points (px) -> LS rotated rect (percent; anchor = top-left before rotation, deg clockwise)."""
    p = np.asarray(pts, float)
    v1, v3 = p[1] - p[0], p[3] - p[0]
    if v1[0] * v3[1] - v1[1] * v3[0] < 0:  # make p0->p1->p2 clockwise in image coords
        p = p[[0, 3, 2, 1]]
        v1, v3 = p[1] - p[0], p[3] - p[0]
    rot = math.degrees(math.atan2(v1[1], v1[0])) % 360
    return dict(x=100 * p[0, 0] / W, y=100 * p[0, 1] / H, width=100 * np.hypot(*v1) / W,
                height=100 * np.hypot(*v3) / H, rotation=rot)


def ls_to_poly(v, W, H):
    x, y, w, h = v["x"] * W / 100, v["y"] * H / 100, v["width"] * W / 100, v["height"] * H / 100
    t = math.radians(v.get("rotation", 0))
    d, n = np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])
    p0 = np.array([x, y])
    return np.stack([p0, p0 + w * d, p0 + w * d + h * n, p0 + h * n])


def add_seed_boxes(pre, idx, seeds_path, size):
    """Project seed polygons (GeoJSON, EPSG:4326) into chips as candidate OBBs; returns seeded chip names."""
    from pyproj import Transformer
    from rasterio.transform import Affine
    from shapely.geometry import Polygon, shape
    tr = Transformer.from_crs("EPSG:4326", idx["crs"], always_xy=True)
    seeds = []
    for f in json.load(open(seeds_path))["features"]:
        ring = [tr.transform(x, y) for x, y in shape(f["geometry"]).exterior.coords]
        seeds.append((ring, f["properties"]))
    seeded = set()
    for chip, meta in idx["chips"].items():
        inv = ~Affine(*meta["transform"])
        for ring, props in seeds:
            px = [inv * p for p in ring]
            cx, cy = np.mean([p[0] for p in px[:-1]]), np.mean([p[1] for p in px[:-1]])
            if 0 <= cx < size and 0 <= cy < size:
                obb = list(Polygon(px).minimum_rotated_rectangle.exterior.coords)[:4]
                pre.setdefault(chip, []).append({"cls": props.get("class", "brick_kiln"), "conf": 1.0,
                                                 "source": "seed", "obb": [list(p) for p in obb]})
                seeded.add(chip)
    return sorted(seeded)


def cmd_tasks(a):
    tiles = pathlib.Path(a.tiles)
    idx = json.load(open(tiles / "index.json"))
    pre = json.load(open(tiles / "prelabels.json")) if (tiles / "prelabels.json").exists() else {}
    chips = list(idx["chips"])
    random.Random(0).shuffle(chips)
    # half the budget from chips with candidates (high-conf first), half random for unbiased negatives
    seeded = add_seed_boxes(pre, idx, a.seeds, a.size) if a.seeds else []
    with_c = sorted([c for c in chips if c in pre and c not in seeded], key=lambda c: -max(d["conf"] for d in pre[c]))
    pick = seeded + with_c[: max(0, a.n // 2 - len(seeded))]
    pick += [c for c in chips if c not in pick][: a.n - len(pick)]
    tasks = []
    for c in pick:
        res = []
        for i, d in enumerate(pre.get(c, [])):
            lab = PRE_MAP.get(d["cls"])
            if lab in OBJ_CLASSES:
                pts = d["obb"] if "obb" in d else [[d["xyxy"][0], d["xyxy"][1]], [d["xyxy"][2], d["xyxy"][1]],
                                                  [d["xyxy"][2], d["xyxy"][3]], [d["xyxy"][0], d["xyxy"][3]]]
                res.append({"id": f"p{i}", "from_name": "obj", "to_name": "image", "type": "rectanglelabels",
                            "value": poly_to_ls(pts, a.size, a.size) | {"rectanglelabels": [lab]}, "score": d["conf"]})
            elif lab in AREA_CLASSES and "xyxy" in d:
                x0, y0, x1, y1 = (100 * v / a.size for v in d["xyxy"])
                res.append({"id": f"p{i}", "from_name": "area", "to_name": "image", "type": "polygonlabels",
                            "value": {"points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "polygonlabels": [lab]},
                            "score": d["conf"]})
        t = {"data": {"image": f"{a.base_url}/{tiles.name}/images/{c}", "chip": c, "tiles": tiles.name}}
        if res:
            t["predictions"] = [{"model_version": "prelabel_v0", "result": res}]
        tasks.append(t)
    out = tiles / "ls_tasks.json"
    json.dump(tasks, open(out, "w"))
    print(f"{len(tasks)} tasks ({sum('predictions' in t for t in tasks)} with candidates) -> {out}")


def cmd_export(a):
    from rasterio.transform import Affine
    tiles = pathlib.Path(a.tiles)
    idx = json.load(open(tiles / "index.json"))
    out = pathlib.Path(a.out)
    (out / "yolo_obb").mkdir(parents=True, exist_ok=True)
    feats = []
    for t in json.load(open(a.ls_json)):
        chip = t["data"]["chip"]
        anns = [x for x in t.get("annotations", []) if not x.get("was_cancelled")]
        if not anns:
            continue
        T = Affine(*idx["chips"][chip]["transform"])
        attrs = {}
        for r in anns[-1]["result"]:
            if r["type"] == "choices":
                attrs.setdefault(r["id"], {})[r["from_name"]] = r["value"]["choices"][0]
        lines = []
        for r in anns[-1]["result"]:
            if r["type"] == "rectanglelabels":
                lab = r["value"]["rectanglelabels"][0]
                poly = ls_to_poly(r["value"], a.size, a.size)
                lines.append(f"{OBJ_CLASSES.index(lab)} " + " ".join(f"{v / a.size:.6f}" for v in poly.ravel()))
            elif r["type"] == "polygonlabels":
                lab = r["value"]["polygonlabels"][0]
                poly = np.array(r["value"]["points"]) * a.size / 100
            else:
                continue
            ring = [list(T * (float(px), float(py))) for px, py in poly]
            feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]},
                          "properties": {"class": lab, "chip": chip, **attrs.get(r["id"], {})}})
        # every reviewed chip gets a label file (empty file = verified negative)
        (out / "yolo_obb" / chip.replace(".jpg", ".txt")).write_text("\n".join(lines))
    gj = {"type": "FeatureCollection", "crs_epsg": idx["crs"], "features": feats}
    json.dump(gj, open(out / "labels_mapcoords.geojson", "w"))
    print(f"{len(feats)} labelled regions; YOLO-OBB -> {out/'yolo_obb'}; map coords ({idx['crs']}) -> {out/'labels_mapcoords.geojson'}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("tasks"); t.add_argument("tiles"); t.add_argument("--base-url", default="http://localhost:8081")
    t.add_argument("--n", type=int, default=300); t.add_argument("--size", type=int, default=1024)
    t.add_argument("--seeds", help="GeoJSON of known objects (e.g. data/labels/kiln_seeds_delhi_kiln_belt.geojson)")
    e = sp.add_parser("export"); e.add_argument("ls_json"); e.add_argument("--tiles", required=True)
    e.add_argument("--out", required=True); e.add_argument("--size", type=int, default=1024)
    a = ap.parse_args()
    {"tasks": cmd_tasks, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
