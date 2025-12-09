
"""
Robust 20k Downloader with Wikipedia Augmentation
-------------------------------------------------
- Collects images + metadata from The Met API (European Paintings + Drawings and Prints)
- Adds Wikipedia summaries for artist/title when available
- Checkpoints every N samples (CSV + JSON lines)
- Resumable: will skip already-downloaded images by checking file existence
Requirements:
  pip install requests pandas tqdm
Optional:
  pip install tenacity  (for robust retries)
"""
import os, json, time, csv
from pathlib import Path
from typing import Dict, Any, Optional
import requests
import pandas as pd
from tqdm import tqdm

print("SCRIPT IS RUNNING!!")


TARGET_COUNT = int(os.environ.get("TARGET_COUNT", "20000"))
SAVE_DIR = Path(os.environ.get("SAVE_DIR", "./met20k"))
IMG_DIR = SAVE_DIR / "images"
METADATA_CSV = SAVE_DIR / "metadata.csv"
METADATA_JSONL = SAVE_DIR / "metadata.jsonl"
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "1000"))
SLEEP_SEC = float(os.environ.get("SLEEP_SEC", "0.35"))

DEPT_URL = "https://collectionapi.metmuseum.org/public/collection/v1/departments"
SEARCH_URL = "https://collectionapi.metmuseum.org/public/collection/v1/search"
OBJECT_URL = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"

WIKI_SEARCH = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

SAVE_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

def safe_json(url: str, params: Optional[dict] = None, timeout=8) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code != 200: return None
        t = r.text.strip()
        if not t or not t.startswith("{"): return None
        return r.json()
    except Exception:
        return None

def wiki_summary(query: str) -> Optional[Dict[str, Any]]:
    if not query: return None
    # 1) search
    s_params = {"action": "query", "list": "search", "srsearch": query, "format": "json", "utf8": 1}
    sr = safe_json(WIKI_SEARCH, params=s_params)
    if not sr or "query" not in sr or not sr["query"]["search"]:
        return None
    title = sr["query"]["search"][0]["title"]
    # 2) summary
    sm = safe_json(WIKI_SUMMARY.format(title.replace(" ", "_")))
    if not sm: return None
    return {
        "wiki_title": title,
        "wiki_description": sm.get("description"),
        "wiki_extract": sm.get("extract"),
        "wiki_url": sm.get("content_urls", {}).get("desktop", {}).get("page"),
    }

def get_department_ids():
    dep = safe_json(DEPT_URL) or {}
    european, drawings = None, None
    for d in dep.get("departments", []):
        name = d.get("displayName", "")
        if "European Paintings" in name:
            european = d.get("departmentId")
        if "Drawings and Prints" in name:
            drawings = d.get("departmentId")
    return european, drawings

def search_ids(dept_id: int):
    params = {"departmentId": dept_id, "hasImages": "true", "q": "*"}
    js = safe_json(SEARCH_URL, params=params) or {}
    return js.get("objectIDs", []) or []

def safe_image_download(url: str, out_path: Path) -> bool:
    try:
        r = requests.get(url, timeout=12)
        if r.status_code != 200: return False
        out_path.write_bytes(r.content)
        return True
    except Exception:
        return False

def load_existing_ids() -> set:
    existing = set()
    for p in IMG_DIR.glob("*.jpg"):
        try:
            existing.add(int(p.stem))
        except Exception:
            continue
    return existing

def append_jsonl(rec: Dict[str, Any]):
    with open(METADATA_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def write_csv_header_if_needed():
    if METADATA_CSV.exists(): return
    with open(METADATA_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "objectID","title","artistDisplayName","artistDisplayBio","objectDate","medium","department",
            "primaryImage","localImagePath","wiki_title","wiki_description","wiki_extract","wiki_url"
        ])

def append_csv_row(rec: Dict[str, Any]):
    with open(METADATA_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            rec.get("objectID"), rec.get("title"), rec.get("artistDisplayName"), rec.get("artistDisplayBio"),
            rec.get("objectDate"), rec.get("medium"), rec.get("department"),
            rec.get("primaryImage"), rec.get("localImagePath"),
            rec.get("wiki_title"), rec.get("wiki_description"), rec.get("wiki_extract"), rec.get("wiki_url")
        ])

def main():
    print("STARTING...")
    dep = safe_json(DEPT_URL)
    print("DEPT", dep)

    write_csv_header_if_needed()
    european, drawings = get_department_ids()
    print("Departments:", european, drawings)
    ids = list(dict.fromkeys(search_ids(european) + search_ids(drawings)))
    print("Total candidate IDs:", len(ids))

    done_ids = load_existing_ids()
    print("Already have images:", len(done_ids))

    success = len(done_ids)
    pbar = tqdm(total=TARGET_COUNT, initial=success, desc="Collected")

    for oid in ids:
        if success >= TARGET_COUNT: break
        if oid in done_ids: 
            pbar.update(1)
            success += 1
            continue

        obj = safe_json(OBJECT_URL.format(oid))
        if not obj:
            time.sleep(SLEEP_SEC)
            continue

        img_url = (obj.get("primaryImage") or "").strip()
        if not img_url:
            time.sleep(SLEEP_SEC)
            continue

        out_path = IMG_DIR / f"{oid}.jpg"
        if not safe_image_download(img_url, out_path):
            time.sleep(SLEEP_SEC)
            continue

        # Wikipedia augmentation
        w_artist = wiki_summary(obj.get("artistDisplayName", ""))
        # Try title if artist fails or to complement
        w_title = wiki_summary(obj.get("title", ""))

        rec = {
            "objectID": obj.get("objectID"),
            "title": obj.get("title"),
            "artistDisplayName": obj.get("artistDisplayName"),
            "artistDisplayBio": obj.get("artistDisplayBio"),
            "objectDate": obj.get("objectDate"),
            "medium": obj.get("medium"),
            "department": obj.get("department"),
            "primaryImage": img_url,
            "localImagePath": str(out_path),
            # Wikipedia merged
            "wiki_title": (w_artist or w_title or {}).get("wiki_title"),
            "wiki_description": (w_artist or w_title or {}).get("wiki_description"),
            "wiki_extract": (w_artist or w_title or {}).get("wiki_extract"),
            "wiki_url": (w_artist or w_title or {}).get("wiki_url"),
        }

        append_jsonl(rec)
        append_csv_row(rec)

        success += 1
        pbar.update(1)
        time.sleep(SLEEP_SEC)

    pbar.close()
    print("Done. Saved to:", METADATA_CSV, "and", METADATA_JSONL, "Images in:", IMG_DIR)

if __name__ == "__main__":
    main()
