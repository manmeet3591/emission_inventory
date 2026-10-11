"""Maxar imagery.

Two routes:
  * open  – Maxar Open Data Program STAC (public, event-based, mostly disaster AOIs; useful for US pretraining).
  * csda  – NASA CSDA-provisioned Maxar access. The exact endpoint/credential flow depends on how CSDA
            onboarded you (check the onboarding email); fill in `csda_search` once confirmed.

    python -m src.data.maxar_download open --list-events
    python -m src.data.maxar_download open --event <collection-id> --aoi us_la_basin
"""
import argparse
import json
import pathlib

import requests
import yaml

OPEN_STAC = "https://maxar-opendata.s3.amazonaws.com/events/catalog.json"
ROOT = pathlib.Path(__file__).resolve().parents[2]


def _get(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def _join(base, href):
    return href if href.startswith("http") else base.rsplit("/", 1)[0] + "/" + href.lstrip("./")


def list_events():
    cat = _get(OPEN_STAC)
    return [(l["href"], l.get("title", "")) for l in cat["links"] if l["rel"] == "child"]


def intersects(b, a):
    return not (b[2] < a[0] or b[0] > a[2] or b[3] < a[1] or b[1] > a[3])


def open_items(event_href, aoi_bbox):
    """Walk an event collection and yield STAC items overlapping the AOI."""
    coll_url = _join(OPEN_STAC, event_href)
    stack = [coll_url]
    while stack:
        url = stack.pop()
        node = _get(url)
        if node.get("type") == "Feature":
            if intersects(node["bbox"], aoi_bbox):
                yield node
            continue
        for l in node.get("links", []):
            if l["rel"] in ("child", "item"):
                stack.append(_join(url, l["href"]))


def csda_search(aoi_bbox, start, end):
    raise NotImplementedError("Fill in once CSDA Maxar access details are confirmed (see onboarding email).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("route", choices=["open", "csda"])
    ap.add_argument("--list-events", action="store_true")
    ap.add_argument("--event")
    ap.add_argument("--aoi")
    a = ap.parse_args()

    if a.route == "open" and a.list_events:
        for href, title in list_events():
            print(href, "|", title)
        return
    cfg = yaml.safe_load(open(ROOT / "configs/aoi.yaml"))
    bbox = cfg["aois"][a.aoi]["bbox"]
    out = ROOT / "data/raw/maxar" / a.aoi
    out.mkdir(parents=True, exist_ok=True)
    if a.route == "open":
        items = list(open_items(a.event, bbox))
        json.dump(items, open(out / "open_items.json", "w"))
        print(f"{len(items)} items; visual COG hrefs:")
        for it in items:
            print(it["assets"].get("visual", {}).get("href"))
    else:
        csda_search(bbox, *cfg["dates"]["default"])


if __name__ == "__main__":
    main()
