"""Create/find a Label Studio project and import ls_tasks.json files (LS_TOKEN = personal access token in .env).

    python -m src.label.ls_push --title "Delhi SkySat pilot" data/tiles/A_rgb data/tiles/B_rgb ...
    python -m src.label.ls_push --title "Delhi SkySat pilot" --export data/labels/ls_export.json
"""
import argparse
import json
import pathlib

import requests

from src.data.planet_download import load_dotenv, ROOT

URL = "http://localhost:8080"


def client():
    import os
    load_dotenv()
    r = requests.post(f"{URL}/api/token/refresh", json={"refresh": os.environ["LS_TOKEN"]})
    r.raise_for_status()
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {r.json()['access']}"
    return s


def project(s, title):
    for p in s.get(f"{URL}/api/projects", params={"page_size": 1000}).json()["results"]:
        if p["title"] == title:
            return p["id"]
    r = s.post(f"{URL}/api/projects", json={
        "title": title, "label_config": (ROOT / "configs/labelstudio.xml").read_text(),
        "show_collab_predictions": True, "description": "Emission-relevant objects; review seeds/prelabels, add missing."})
    r.raise_for_status()
    return r.json()["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tiles", nargs="*")
    ap.add_argument("--title", required=True)
    ap.add_argument("--export")
    a = ap.parse_args()
    s = client()
    pid = project(s, a.title)
    if a.export:
        r = s.get(f"{URL}/api/projects/{pid}/export", params={"exportType": "JSON"})
        r.raise_for_status()
        pathlib.Path(a.export).write_bytes(r.content)
        print(f"exported {len(r.json())} tasks -> {a.export}")
        return
    for t in a.tiles:
        tasks = json.load(open(pathlib.Path(t) / "ls_tasks.json"))
        r = s.post(f"{URL}/api/projects/{pid}/import", json=tasks)
        r.raise_for_status()
        print(f"project {pid}: +{r.json().get('task_count', len(tasks))} tasks from {pathlib.Path(t).name}")
    print(f"open {URL}/projects/{pid}/data")


if __name__ == "__main__":
    main()
