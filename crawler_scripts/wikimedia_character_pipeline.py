#!/usr/bin/env python3
"""
Wikimedia Commons cosplay-by-character pipeline.

Stage 0: List all direct subcategories of Category:Cosplay by character.
Stage 1: Recursively count images per character using categoryinfo (fast, no file listing).
         Characters with >= MIN_IMAGES are marked qualifying.
Stage 2: For each qualifying character, discover subcategories → collect metadata → download images.
         Writes <OUT_DIR>/qualifying_characters.json for the Reddit pipeline.

All stages are resumable. Run again after interruption to continue.
"""
import csv
import html
import json
import random
import re
import time
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests

# ── Config ──────────────────────────────────────────────────────────────────

CHARACTER_ROOT = "Category:Cosplay by character"
MIN_IMAGES = 50

OUT_DIR = Path("data/characters")
PIPELINE_STATE_PATH = OUT_DIR / "pipeline_state.json"
QUALIFYING_CHARS_PATH = OUT_DIR / "qualifying_characters.json"

USER_AGENT = "CostumeRecognitionBot/0.1 (contact: rha.kempf@gmail.com)"

CATEGORY_BATCH_SIZE = 500
FILEINFO_BATCH_SIZE = 25
MEDIAINFO_BATCH_SIZE = 50
WIKIDATA_LABEL_BATCH_SIZE = 50

API_PAUSE_SECONDS = (1.5, 3.0)
DOWNLOAD_PAUSE_SECONDS = (2.5, 5.0)
MAX_RETRIES = 6
REQUEST_TIMEOUT = 120
MAXLAG = 5

API = "https://commons.wikimedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})

# ── Utilities ────────────────────────────────────────────────────────────────


def sleep_range(bounds):
    time.sleep(random.uniform(bounds[0], bounds[1]))


def clean_html_text(value: str) -> str:
    if not value:
        return ""
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_category_title(title: str) -> str:
    title = title.strip().replace("_", " ")
    if not title.startswith("Category:"):
        title = "Category:" + title
    return title


def safe_filename(name: str) -> str:
    name = unquote(name)
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    return name.strip()


def filename_from_url(url: str) -> str:
    path = urlparse(url).path
    return safe_filename(path.split("/")[-1])


def character_slug(category_title: str) -> str:
    """'Category:Cosplay of Hu Tao' → 'hu_tao'"""
    title = normalize_category_title(category_title)
    name = title.removeprefix("Category:Cosplay of ").strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    return name.lower().strip("_") or "unknown"


def character_name_from_category(category_title: str) -> str:
    """'Category:Cosplay of Hu Tao' → 'Hu Tao'"""
    title = normalize_category_title(category_title)
    return title.removeprefix("Category:Cosplay of ").strip()


# ── HTTP layer ───────────────────────────────────────────────────────────────


def request_with_retry(url, *, params=None, data=None, stream=False,
                       timeout=REQUEST_TIMEOUT, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        if data is not None:
            resp = session.post(url, params=params, data=data, stream=stream, timeout=timeout)
        else:
            resp = session.get(url, params=params, stream=stream, timeout=timeout)

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = int(retry_after) if (retry_after and retry_after.isdigit()) else min(60 * (2 ** attempt), 1800)
            print(f"429 Too Many Requests. Sleeping {wait}s …")
            resp.close()
            time.sleep(wait)
            continue

        if resp.status_code == 503 and not stream:
            try:
                err = resp.json().get("error", {})
                if err.get("code") == "maxlag":
                    wait = min(10 * (attempt + 1), 60)
                    print(f"maxlag hit. Sleeping {wait}s …")
                    resp.close()
                    time.sleep(wait)
                    continue
            except Exception:
                pass

        resp.raise_for_status()
        return resp

    raise RuntimeError(f"Failed after {max_retries} retries: {url}")


def api_get(params: dict) -> dict:
    merged = {"format": "json", "maxlag": MAXLAG}
    merged.update(params)
    resp = request_with_retry(API, params=merged)
    data = resp.json()
    if data.get("error", {}).get("code") == "maxlag":
        raise RuntimeError(f"maxlag: {data['error']}")
    return data


def api_post(params: dict) -> dict:
    merged = {"format": "json", "maxlag": MAXLAG}
    merged.update(params)
    resp = request_with_retry(API, data=merged)
    data = resp.json()
    if data.get("error", {}).get("code") == "maxlag":
        raise RuntimeError(f"maxlag: {data['error']}")
    return data


def wikidata_post(params: dict) -> dict:
    merged = {"format": "json"}
    merged.update(params)
    resp = request_with_retry(WIKIDATA_API, data=merged)
    return resp.json()


# ── Commons API helpers ───────────────────────────────────────────────────────


def get_category_members(category_title: str, cmtype: str):
    """Yields categorymembers rows (paginated)."""
    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category_title,
            "cmtype": cmtype,
            "cmlimit": CATEGORY_BATCH_SIZE,
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = api_get(params)
        for member in data.get("query", {}).get("categorymembers", []):
            yield member
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
        sleep_range(API_PAUSE_SECONDS)


def get_category_direct_file_count(category_title: str) -> int:
    """Returns the number of files directly in a category (non-recursive)."""
    params = {
        "action": "query",
        "prop": "categoryinfo",
        "titles": category_title,
    }
    data = api_get(params)
    for page in data.get("query", {}).get("pages", {}).values():
        return page.get("categoryinfo", {}).get("files", 0)
    return 0


def get_file_pages_info(file_titles):
    all_pages = []
    for start in range(0, len(file_titles), FILEINFO_BATCH_SIZE):
        batch = file_titles[start: start + FILEINFO_BATCH_SIZE]
        params = {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "info|categories|imageinfo",
            "inprop": "url",
            "cllimit": "max",
            "iiprop": "url|extmetadata",
        }
        data = api_get(params)
        all_pages.extend(data.get("query", {}).get("pages", {}).values())
        sleep_range(API_PAUSE_SECONDS)
    return all_pages


def get_mediainfo_entities(m_ids):
    all_entities = {}
    for start in range(0, len(m_ids), MEDIAINFO_BATCH_SIZE):
        batch = m_ids[start: start + MEDIAINFO_BATCH_SIZE]
        data = api_post({
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": "labels|descriptions|claims",
        })
        all_entities.update(data.get("entities", {}))
        sleep_range(API_PAUSE_SECONDS)
    return all_entities


def get_wikidata_labels(qids, lang="en"):
    result = {}
    unique_qids = sorted(set(qids))
    for start in range(0, len(unique_qids), WIKIDATA_LABEL_BATCH_SIZE):
        batch = unique_qids[start: start + WIKIDATA_LABEL_BATCH_SIZE]
        entities = wikidata_post({
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "languages": lang,
            "props": "labels",
        }).get("entities", {})
        for qid, ent in entities.items():
            result[qid] = ent.get("labels", {}).get(lang, {}).get("value", "")
        sleep_range(API_PAUSE_SECONDS)
    return result


def extract_extmetadata(extmeta):
    def get(name):
        return clean_html_text((extmeta.get(name) or {}).get("value", ""))
    return {
        "license_short": get("LicenseShortName"),
        "license_url": get("LicenseUrl"),
        "artist": get("Artist"),
        "credit": get("Credit"),
        "usage_terms": get("UsageTerms"),
        "image_description": get("ImageDescription"),
        "object_name": get("ObjectName"),
    }


def extract_categories(page):
    cats = []
    for c in page.get("categories", []):
        title = c.get("title", "")
        if title.startswith("Category:"):
            title = title[len("Category:"):]
        cats.append(title)
    return cats


def extract_p180_qids(entity):
    out = []
    for stmt in entity.get("claims", {}).get("P180", []):
        value = stmt.get("mainsnak", {}).get("datavalue", {}).get("value", {})
        if isinstance(value, dict) and "id" in value:
            out.append(value["id"])
    return out


def build_costume_hint(depicts_labels, categories):
    if depicts_labels:
        return "; ".join(depicts_labels)
    costumeish = [
        c for c in categories
        if any(k in c.lower() for k in ["cosplay", "costume", "character", "anime", "manga", "fiction", "comic", "game"])
    ]
    return "; ".join(costumeish[:10])


# ── Per-character download helpers ──────────────────────────────────────────


def load_existing_records(jsonl_path: Path):
    records, by_pageid, seen_pageids = [], {}, set()
    if not jsonl_path.exists():
        return records, by_pageid, seen_pageids
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
            pid = rec.get("pageid")
            if pid is not None:
                by_pageid[pid] = rec
                seen_pageids.add(pid)
    return records, by_pageid, seen_pageids


def save_records_jsonl(records, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_csv(records, path: Path):
    fieldnames = [
        "pageid", "mediainfo_id", "title", "file_page_url", "image_url",
        "local_image_path", "is_downloaded", "download_status", "download_attempts",
        "last_error", "costume_hint", "depicts_qids", "depicts_labels", "categories",
        "source_category", "root_category", "image_description", "object_name",
        "artist", "credit", "license_short", "license_url", "usage_terms",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            row = row.copy()
            row["depicts_qids"] = "|".join(row.get("depicts_qids", []))
            row["depicts_labels"] = "|".join(row.get("depicts_labels", []))
            row["categories"] = "|".join(row.get("categories", []))
            writer.writerow(row)


def refresh_download_flags(records):
    for rec in records:
        local_path = Path(rec["local_image_path"])
        exists = local_path.exists()
        rec["is_downloaded"] = exists
        if exists:
            rec["download_status"] = "downloaded"
            rec["last_error"] = ""
    return records


def download_image(url: str, dest: Path):
    if dest.exists():
        return "already_exists"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(MAX_RETRIES):
        sleep_range(DOWNLOAD_PAUSE_SECONDS)
        resp = session.get(url, stream=True, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = int(retry_after) if (retry_after and retry_after.isdigit()) else min(60 * (2 ** attempt), 1800)
            print(f"429 on download. Sleeping {wait}s …")
            resp.close()
            time.sleep(wait)
            continue
        try:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return "downloaded"
        finally:
            resp.close()
    raise RuntimeError(f"Too many retries: {url}")


# ── Pipeline state ───────────────────────────────────────────────────────────


def load_pipeline_state() -> dict:
    if PIPELINE_STATE_PATH.exists():
        with open(PIPELINE_STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {
        "stage0_complete": False,
        "character_categories": [],
        "stage1_complete": False,
        "image_counts": {},
        "qualifying_characters": [],
        "downloaded_characters": [],
    }


def save_pipeline_state(state: dict):
    PIPELINE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PIPELINE_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ── Stage 0: discover character categories ───────────────────────────────────


def stage0_discover_character_categories(state: dict) -> dict:
    if state.get("stage0_complete"):
        print(f"Stage 0 already complete ({len(state['character_categories'])} character categories).")
        return state

    print(f"\n{'='*70}")
    print(f"Stage 0: Discovering character categories under {CHARACTER_ROOT}")
    print(f"{'='*70}")

    found = set(state.get("character_categories", []))

    for member in get_category_members(CHARACTER_ROOT, "subcat"):
        title = normalize_category_title(member.get("title", ""))
        if title and title not in found:
            found.add(title)
            print(f"  Found: {title}")

        sleep_range((0.1, 0.3))

    state["character_categories"] = sorted(found)
    state["stage0_complete"] = True
    save_pipeline_state(state)

    print(f"\nStage 0 complete. Found {len(found)} character categories.")
    return state


# ── Stage 1: count images recursively per character ──────────────────────────


def count_images_recursive(category_title: str) -> int:
    """
    BFS over the category tree starting at category_title.
    Uses categoryinfo API (one call per category) to count direct files,
    avoiding the need to list every file individually.
    """
    visited: set[str] = set()
    pending = [normalize_category_title(category_title)]
    total = 0

    while pending:
        current = pending.pop(0)
        if current in visited:
            continue
        visited.add(current)

        count = get_category_direct_file_count(current)
        total += count
        sleep_range(API_PAUSE_SECONDS)

        for member in get_category_members(current, "subcat"):
            subcat = normalize_category_title(member.get("title", ""))
            if subcat and subcat not in visited:
                pending.append(subcat)
        sleep_range(API_PAUSE_SECONDS)

    return total


def stage1_count_and_filter(state: dict) -> dict:
    if state.get("stage1_complete"):
        q = state.get("qualifying_characters", [])
        print(f"Stage 1 already complete ({len(q)} qualifying characters).")
        return state

    print(f"\n{'='*70}")
    print(f"Stage 1: Counting images per character (threshold >= {MIN_IMAGES})")
    print(f"{'='*70}")

    # Only process real character categories; skip structural meta-categories
    # like "by country", "by event", "by performer", etc.
    character_categories = [
        c for c in state["character_categories"]
        if normalize_category_title(c).startswith("Category:Cosplay of ")
    ]
    skipped = len(state["character_categories"]) - len(character_categories)
    if skipped:
        print(f"Skipping {skipped} non-character meta-categories (no 'Cosplay of' prefix).\n")
    image_counts: dict = state.get("image_counts", {})

    for i, cat in enumerate(character_categories, start=1):
        if cat in image_counts:
            print(f"[{i}/{len(character_categories)}] SKIP (cached) {cat} = {image_counts[cat]}")
            continue

        print(f"[{i}/{len(character_categories)}] Counting: {cat} …", end=" ", flush=True)
        count = count_images_recursive(cat)
        image_counts[cat] = count
        print(f"{count} images")

        state["image_counts"] = image_counts
        save_pipeline_state(state)

    qualifying = [cat for cat, cnt in image_counts.items() if cnt >= MIN_IMAGES]
    qualifying.sort()

    state["image_counts"] = image_counts
    state["qualifying_characters"] = qualifying
    state["stage1_complete"] = True
    save_pipeline_state(state)

    with open(QUALIFYING_CHARS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [
                {"category": cat, "character_name": character_name_from_category(cat),
                 "image_count": image_counts[cat], "slug": character_slug(cat)}
                for cat in qualifying
            ],
            f, ensure_ascii=False, indent=2,
        )

    print(f"\nStage 1 complete. {len(qualifying)} characters qualify (>= {MIN_IMAGES} images).")
    print(f"Qualifying list written to: {QUALIFYING_CHARS_PATH}")
    return state


# ── Stage 2: download images for qualifying characters ───────────────────────


def download_character(root_category: str, char_out_dir: Path):
    """
    Full pipeline for one character category:
      sub-stage A: discover all subcategories (BFS)
      sub-stage B: collect file metadata
      sub-stage C: download images
    State is saved inside char_out_dir/metadata/.
    """
    meta_dir = char_out_dir / "metadata"
    images_dir = char_out_dir / "images"
    meta_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    state_path = meta_dir / "crawl_state.json"
    jsonl_path = meta_dir / "metadata.jsonl"
    csv_path = meta_dir / "metadata.csv"
    failed_path = meta_dir / "failed_downloads.jsonl"
    subcats_path = meta_dir / "all_subcategories.txt"

    root_category = normalize_category_title(root_category)

    # Load or init per-character state
    if state_path.exists():
        with open(state_path, encoding="utf-8") as f:
            char_state = json.load(f)
    else:
        char_state = {
            "root_category": root_category,
            "subcat_discovery_complete": False,
            "metadata_complete": False,
            "pending_categories": [root_category],
            "visited_categories": [],
            "discovered_categories": [root_category],
            "pending_metadata_categories": [],
            "processed_metadata_categories": [],
        }

    def _save_char_state():
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(char_state, f, ensure_ascii=False, indent=2)

    # Sub-stage A: discover subcategories
    if not char_state.get("subcat_discovery_complete"):
        visited = set(char_state.get("visited_categories", []))
        discovered = set(char_state.get("discovered_categories", [root_category]))
        pending = list(char_state.get("pending_categories", [root_category]))

        while pending:
            current = pending.pop(0)
            if current in visited:
                continue
            print(f"    [subcat] {current}")
            for member in get_category_members(current, "subcat"):
                subcat = normalize_category_title(member.get("title", ""))
                if subcat and subcat not in discovered:
                    discovered.add(subcat)
                    pending.append(subcat)
            visited.add(current)
            char_state.update({
                "pending_categories": pending,
                "visited_categories": sorted(visited),
                "discovered_categories": sorted(discovered),
            })
            _save_char_state()
            sleep_range(API_PAUSE_SECONDS)

        char_state["subcat_discovery_complete"] = True
        char_state["pending_metadata_categories"] = sorted(discovered)
        _save_char_state()

        with open(subcats_path, "w", encoding="utf-8") as f:
            for cat in sorted(discovered):
                f.write(cat + "\n")

    # Sub-stage B: collect metadata
    if not char_state.get("metadata_complete"):
        records, by_pageid, seen_pageids = load_existing_records(jsonl_path)
        records = refresh_download_flags(records)

        pending_meta = list(char_state.get("pending_metadata_categories", []))
        processed_meta = set(char_state.get("processed_metadata_categories", []))
        new_records = 0

        while pending_meta:
            cat_title = pending_meta.pop(0)
            if cat_title in processed_meta:
                continue

            file_titles = [
                m.get("title", "") for m in get_category_members(cat_title, "file")
                if m.get("title", "").startswith("File:")
            ]
            print(f"    [meta] {cat_title} — {len(file_titles)} files")

            pages = get_file_pages_info(file_titles)
            page_records = [
                (p.get("pageid"), f"M{p.get('pageid')}", p)
                for p in pages
                if p.get("pageid") and p["pageid"] not in seen_pageids
            ]

            m_ids = [mid for _, mid, _ in page_records]
            mediainfo = get_mediainfo_entities(m_ids)
            sleep_range(API_PAUSE_SECONDS)

            all_qids = []
            p180_by_mid = {}
            for _, mid, _ in page_records:
                qids = extract_p180_qids(mediainfo.get(mid, {}))
                p180_by_mid[mid] = qids
                all_qids.extend(qids)

            qid_to_label = get_wikidata_labels(all_qids)
            sleep_range(API_PAUSE_SECONDS)

            for pageid, mid, page in page_records:
                imageinfo = (page.get("imageinfo") or [{}])[0]
                image_url = imageinfo.get("url", "")
                if not image_url:
                    continue

                ext = extract_extmetadata(imageinfo.get("extmetadata", {}))
                categories = extract_categories(page)
                depicts_qids = p180_by_mid.get(mid, [])
                depicts_labels = [qid_to_label.get(q, q) for q in depicts_qids]
                local_path = images_dir / filename_from_url(image_url)

                record = {
                    "pageid": pageid,
                    "mediainfo_id": mid,
                    "title": page.get("title", ""),
                    "file_page_url": page.get("fullurl", ""),
                    "image_url": image_url,
                    "local_image_path": str(local_path),
                    "is_downloaded": local_path.exists(),
                    "download_status": "downloaded" if local_path.exists() else "pending",
                    "download_attempts": 0,
                    "last_error": "",
                    "costume_hint": build_costume_hint(depicts_labels, categories),
                    "depicts_qids": depicts_qids,
                    "depicts_labels": depicts_labels,
                    "categories": categories,
                    "source_category": cat_title,
                    "root_category": root_category,
                    **ext,
                }
                records.append(record)
                by_pageid[pageid] = record
                seen_pageids.add(pageid)
                new_records += 1

            processed_meta.add(cat_title)
            save_records_jsonl(records, jsonl_path)
            write_csv(records, csv_path)
            char_state["pending_metadata_categories"] = pending_meta
            char_state["processed_metadata_categories"] = sorted(processed_meta)
            char_state["metadata_complete"] = len(pending_meta) == 0
            _save_char_state()
            sleep_range(API_PAUSE_SECONDS)

        print(f"    Metadata complete. {new_records} new records.")

    # Sub-stage C: download images
    records, _, _ = load_existing_records(jsonl_path)
    records = refresh_download_flags(records)
    candidates = [
        r for r in records
        if not r.get("is_downloaded") and r.get("download_status") in ("pending", "failed")
    ]

    downloaded = skipped = failed = 0
    with open(failed_path, "a", encoding="utf-8") as fail_f:
        for idx, rec in enumerate(candidates, start=1):
            url = rec["image_url"]
            dest = Path(rec["local_image_path"])
            rec["download_attempts"] = rec.get("download_attempts", 0) + 1
            try:
                status = download_image(url, dest)
                rec["is_downloaded"] = True
                rec["download_status"] = "downloaded"
                rec["last_error"] = ""
                if status == "already_exists":
                    skipped += 1
                else:
                    downloaded += 1
                print(f"    [{idx}] {status.upper()} → {dest.name}")
            except Exception as e:
                rec["is_downloaded"] = False
                rec["download_status"] = "failed"
                rec["last_error"] = str(e)
                failed += 1
                fail_f.write(json.dumps({"url": url, "error": str(e)}, ensure_ascii=False) + "\n")
                fail_f.flush()
                print(f"    [{idx}] FAILED {url} — {e}")

            save_records_jsonl(records, jsonl_path)
            write_csv(records, csv_path)

    print(f"    Downloads: {downloaded} new, {skipped} existed, {failed} failed.")


def stage2_download_qualifying(state: dict) -> dict:
    qualifying = state.get("qualifying_characters", [])
    if not qualifying:
        print("No qualifying characters to download.")
        return state

    downloaded_set = set(state.get("downloaded_characters", []))

    print(f"\n{'='*70}")
    print(f"Stage 2: Downloading images for {len(qualifying)} qualifying characters")
    print(f"{'='*70}")

    for i, cat in enumerate(qualifying, start=1):
        if cat in downloaded_set:
            print(f"[{i}/{len(qualifying)}] SKIP (done) {cat}")
            continue

        slug = character_slug(cat)
        char_dir = OUT_DIR / slug / "wikimedia"
        print(f"\n[{i}/{len(qualifying)}] {cat} → {char_dir}")

        try:
            download_character(cat, char_dir)
            downloaded_set.add(cat)
            state["downloaded_characters"] = sorted(downloaded_set)
            save_pipeline_state(state)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  ERROR for {cat}: {e}")

    print(f"\nStage 2 complete.")
    return state


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state = load_pipeline_state()

    state = stage0_discover_character_categories(state)
    state = stage1_count_and_filter(state)
    state = stage2_download_qualifying(state)

    print(f"\nPipeline complete.")
    print(f"Qualifying characters JSON: {QUALIFYING_CHARS_PATH}")
    print(f"Data directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
