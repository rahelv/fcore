#!/usr/bin/env python3
import csv
import html
import json
import random
import re
import time
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests


API = "https://commons.wikimedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

ROOT_CATEGORY = "Category:Cosplay"
USER_AGENT = "CostumeRecognitionBot/0.1 (contact: rha.kempf@gmail.com)"

OUT_DIR = Path("/home/ubuntu/data2")
METADATA_DIR = OUT_DIR / "metadata"
IMAGES_DIR = OUT_DIR / "images"
METADATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

JSONL_PATH = METADATA_DIR / "metadata.jsonl"
CSV_PATH = METADATA_DIR / "metadata.csv"
FAILED_DOWNLOADS_PATH = METADATA_DIR / "failed_downloads.jsonl"
STATE_PATH = METADATA_DIR / "crawl_state.json"
SUBCATEGORIES_TXT_PATH = METADATA_DIR / "all_subcategories.txt"

CATEGORY_BATCH_SIZE = 500
FILEINFO_BATCH_SIZE = 25
MEDIAINFO_BATCH_SIZE = 50
WIKIDATA_LABEL_BATCH_SIZE = 50

API_PAUSE_SECONDS = (1.5, 3.0)
DOWNLOAD_PAUSE_SECONDS = (2.5, 5.0)
MAX_RETRIES = 6
REQUEST_TIMEOUT = 120
MAXLAG = 5

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


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


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {
            "root_category": ROOT_CATEGORY,
            "category_discovery_complete": False,
            "metadata_complete": False,
            "pending_categories_to_visit": [ROOT_CATEGORY],
            "visited_categories": [],
            "discovered_categories": [ROOT_CATEGORY],
            "pending_metadata_categories": [],
            "processed_metadata_categories": [],
        }
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def request_with_retry(
    url: str,
    *,
    params=None,
    data=None,
    stream: bool = False,
    timeout: int = REQUEST_TIMEOUT,
    max_retries: int = MAX_RETRIES,
):
    for attempt in range(max_retries):
        if data is not None:
            response = session.post(
                url, params=params, data=data, stream=stream, timeout=timeout
            )
        else:
            response = session.get(url, params=params, stream=stream, timeout=timeout)

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait = int(retry_after)
            else:
                wait = min(60 * (2**attempt), 1800)
            print(f"429 Too Many Requests. Sleeping {wait}s and retrying...")
            response.close()
            time.sleep(wait)
            continue

        if response.status_code == 503 and not stream:
            try:
                data_json = response.json()
                error = data_json.get("error", {})
                if error.get("code") == "maxlag":
                    wait = min(10 * (attempt + 1), 60)
                    print(f"maxlag hit. Sleeping {wait}s and retrying...")
                    response.close()
                    time.sleep(wait)
                    continue
            except Exception:
                pass

        response.raise_for_status()
        return response

    raise RuntimeError(f"Failed after {max_retries} retries: {url}")


def api_get(params: dict) -> dict:
    merged = {"format": "json", "maxlag": MAXLAG}
    merged.update(params)
    response = request_with_retry(API, params=merged, stream=False)
    data = response.json()

    error = data.get("error")
    if error and error.get("code") == "maxlag":
        raise RuntimeError(f"MediaWiki maxlag response: {error}")

    return data


def api_post(params: dict) -> dict:
    merged = {"format": "json", "maxlag": MAXLAG}
    merged.update(params)
    response = request_with_retry(API, data=merged, stream=False)
    data = response.json()

    error = data.get("error")
    if error and error.get("code") == "maxlag":
        raise RuntimeError(f"MediaWiki maxlag response: {error}")

    return data


def wikidata_post(params: dict) -> dict:
    merged = {"format": "json"}
    merged.update(params)
    response = request_with_retry(WIKIDATA_API, data=merged, stream=False)
    return response.json()


def get_category_members(category_title: str, cmtype: str):
    """
    cmtype: 'subcat' or 'file'
    Yields categorymembers rows with pagination.
    """
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
        members = data.get("query", {}).get("categorymembers", [])
        for member in members:
            yield member

        cont = data.get("continue", {})
        cmcontinue = cont.get("cmcontinue")
        if not cmcontinue:
            break

        sleep_range(API_PAUSE_SECONDS)


def get_file_pages_info(file_titles):
    """
    Batch query File:... pages with info|categories|imageinfo.
    """
    if not file_titles:
        return []

    all_pages = []

    for start in range(0, len(file_titles), FILEINFO_BATCH_SIZE):
        batch = file_titles[start : start + FILEINFO_BATCH_SIZE]
        params = {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "info|categories|imageinfo",
            "inprop": "url",
            "cllimit": "max",
            "iiprop": "url|extmetadata",
        }
        data = api_get(params)
        pages = data.get("query", {}).get("pages", {})
        all_pages.extend(pages.values())
        sleep_range(API_PAUSE_SECONDS)

    return all_pages


def get_mediainfo_entities(m_ids):
    if not m_ids:
        return {}

    all_entities = {}

    for start in range(0, len(m_ids), MEDIAINFO_BATCH_SIZE):
        batch = m_ids[start : start + MEDIAINFO_BATCH_SIZE]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": "labels|descriptions|claims",
        }
        data = api_post(params)
        all_entities.update(data.get("entities", {}))
        sleep_range(API_PAUSE_SECONDS)

    return all_entities


def get_wikidata_labels(qids, lang="en"):
    if not qids:
        return {}

    result = {}
    unique_qids = sorted(set(qids))

    for start in range(0, len(unique_qids), WIKIDATA_LABEL_BATCH_SIZE):
        batch = unique_qids[start : start + WIKIDATA_LABEL_BATCH_SIZE]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "languages": lang,
            "props": "labels",
        }
        entities = wikidata_post(params).get("entities", {})
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
            title = title[len("Category:") :]
        cats.append(title)
    return cats


def extract_p180_qids(entity):
    claims = entity.get("claims", {})
    out = []
    for stmt in claims.get("P180", []):
        mainsnak = stmt.get("mainsnak", {})
        datavalue = mainsnak.get("datavalue", {})
        value = datavalue.get("value", {})
        if isinstance(value, dict) and "id" in value:
            out.append(value["id"])
    return out


def build_costume_hint(depicts_labels, categories):
    if depicts_labels:
        return "; ".join(depicts_labels)

    costumeish = [
        c
        for c in categories
        if any(
            k in c.lower()
            for k in [
                "cosplay",
                "costume",
                "character",
                "anime",
                "manga",
                "fiction",
                "comic",
                "game",
            ]
        )
    ]
    return "; ".join(costumeish[:10])


def load_existing_records(jsonl_path: Path):
    records = []
    by_pageid = {}
    seen_pageids = set()

    if not jsonl_path.exists():
        return records, by_pageid, seen_pageids

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
            pageid = rec.get("pageid")
            if pageid is not None:
                by_pageid[pageid] = rec
                seen_pageids.add(pageid)

    return records, by_pageid, seen_pageids


def save_all_records_jsonl(records, jsonl_path: Path):
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_csv_from_records(records, csv_path: Path):
    fieldnames = [
        "pageid",
        "mediainfo_id",
        "title",
        "file_page_url",
        "image_url",
        "local_image_path",
        "is_downloaded",
        "download_status",
        "download_attempts",
        "last_error",
        "costume_hint",
        "depicts_qids",
        "depicts_labels",
        "categories",
        "source_category",
        "root_category",
        "image_description",
        "object_name",
        "artist",
        "credit",
        "license_short",
        "license_url",
        "usage_terms",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
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


def write_subcategories_txt(categories):
    with open(SUBCATEGORIES_TXT_PATH, "w", encoding="utf-8") as f:
        for cat in sorted(set(categories)):
            f.write(cat + "\n")


def stage0_discover_categories(root_category: str, resume: bool = True):
    root_category = normalize_category_title(root_category)

    state = load_state()
    if not resume or state.get("root_category") != root_category:
        state = {
            "root_category": root_category,
            "category_discovery_complete": False,
            "metadata_complete": False,
            "pending_categories_to_visit": [root_category],
            "visited_categories": [],
            "discovered_categories": [root_category],
            "pending_metadata_categories": [],
            "processed_metadata_categories": [],
        }

    visited = set(state.get("visited_categories", []))
    discovered = set(state.get("discovered_categories", [root_category]))
    pending = list(state.get("pending_categories_to_visit", [root_category]))

    print(f"Starting category discovery from root: {root_category}")

    while pending:
        current = pending.pop(0)
        if current in visited:
            continue

        print(f"[CATEGORY] {current}")

        new_subcats = 0
        for member in get_category_members(current, "subcat"):
            subcat = normalize_category_title(member.get("title", ""))
            if not subcat:
                continue
            if subcat not in discovered:
                discovered.add(subcat)
                pending.append(subcat)
                new_subcats += 1

        print(f"    new subcategories found: {new_subcats}")

        visited.add(current)
        state["pending_categories_to_visit"] = pending
        state["visited_categories"] = sorted(visited)
        state["discovered_categories"] = sorted(discovered)
        save_state(state)

        sleep_range(API_PAUSE_SECONDS)

    state["category_discovery_complete"] = True
    state["pending_categories_to_visit"] = []
    state["visited_categories"] = sorted(visited)
    state["discovered_categories"] = sorted(discovered)
    state["pending_metadata_categories"] = sorted(discovered)
    save_state(state)

    write_subcategories_txt(sorted(discovered))

    print("\nCategory discovery complete.")
    print(f"Total categories found (including root): {len(discovered)}")
    print(f"Subcategory list written to: {SUBCATEGORIES_TXT_PATH}")


def stage1_collect_metadata_from_categories(
    root_category: str,
    max_categories: int | None = None,
    resume: bool = True,
):
    root_category = normalize_category_title(root_category)

    state = load_state()
    if not state.get("category_discovery_complete"):
        raise RuntimeError(
            "Category discovery not complete. Run stage0_discover_categories() first."
        )

    records, by_pageid, seen_pageids = load_existing_records(JSONL_PATH)
    records = refresh_download_flags(records)

    if not resume:
        state["pending_metadata_categories"] = state.get("discovered_categories", [])
        state["processed_metadata_categories"] = []

    pending_categories = list(state.get("pending_metadata_categories", []))
    processed_categories = set(state.get("processed_metadata_categories", []))

    categories_done_this_run = 0
    new_records = 0

    print(f"Starting metadata crawl from {len(pending_categories)} categories")

    while pending_categories:
        if max_categories is not None and categories_done_this_run >= max_categories:
            break

        category_title = pending_categories.pop(0)
        if category_title in processed_categories:
            continue

        print(f"\n[METADATA CATEGORY] {category_title}")

        file_titles = []
        for member in get_category_members(category_title, "file"):
            title = member.get("title", "")
            if title.startswith("File:"):
                file_titles.append(title)

        print(f"    files in category: {len(file_titles)}")

        page_records = []
        pages = get_file_pages_info(file_titles)

        for page in pages:
            pageid = page.get("pageid")
            if not pageid or pageid in seen_pageids:
                continue

            mid = f"M{pageid}"
            page_records.append((pageid, mid, page))

        print(f"    new files after dedupe: {len(page_records)}")

        m_ids = [mid for _, mid, _ in page_records]
        mediainfo = get_mediainfo_entities(m_ids)
        sleep_range(API_PAUSE_SECONDS)

        all_qids = []
        p180_by_mid = {}
        for _, mid, _ in page_records:
            entity = mediainfo.get(mid, {})
            qids = extract_p180_qids(entity)
            p180_by_mid[mid] = qids
            all_qids.extend(qids)

        qid_to_label = get_wikidata_labels(all_qids, lang="en")
        sleep_range(API_PAUSE_SECONDS)

        for pageid, mid, page in page_records:
            title = page.get("title", "")
            fullurl = page.get("fullurl", "")
            imageinfo = (page.get("imageinfo") or [{}])[0]
            extmeta = imageinfo.get("extmetadata", {})
            image_url = imageinfo.get("url", "")

            if not image_url:
                print(f"SKIP {title} -> no image URL")
                continue

            ext = extract_extmetadata(extmeta)
            categories = extract_categories(page)
            depicts_qids = p180_by_mid.get(mid, [])
            depicts_labels = [qid_to_label.get(qid, qid) for qid in depicts_qids]
            costume_hint = build_costume_hint(depicts_labels, categories)

            filename = filename_from_url(image_url)
            local_path = IMAGES_DIR / filename
            already_downloaded = local_path.exists()

            record = {
                "pageid": pageid,
                "mediainfo_id": mid,
                "title": title,
                "file_page_url": fullurl,
                "image_url": image_url,
                "local_image_path": str(local_path),
                "is_downloaded": already_downloaded,
                "download_status": "downloaded" if already_downloaded else "pending",
                "download_attempts": 0,
                "last_error": "",
                "costume_hint": costume_hint,
                "depicts_qids": depicts_qids,
                "depicts_labels": depicts_labels,
                "categories": categories,
                "source_category": category_title,
                "root_category": root_category,
                "image_description": ext["image_description"],
                "object_name": ext["object_name"],
                "artist": ext["artist"],
                "credit": ext["credit"],
                "license_short": ext["license_short"],
                "license_url": ext["license_url"],
                "usage_terms": ext["usage_terms"],
            }

            records.append(record)
            by_pageid[pageid] = record
            seen_pageids.add(pageid)
            new_records += 1

            print(
                f"[NEW {new_records}] METADATA -> {title}\n"
                f"    source_category: {category_title}\n"
                f"    downloaded: {already_downloaded}\n"
                f"    costume_hint: {costume_hint or '(none)'}"
            )

        processed_categories.add(category_title)
        categories_done_this_run += 1

        save_all_records_jsonl(records, JSONL_PATH)
        write_csv_from_records(records, CSV_PATH)

        state["pending_metadata_categories"] = pending_categories
        state["processed_metadata_categories"] = sorted(processed_categories)
        state["metadata_complete"] = len(pending_categories) == 0
        save_state(state)

        sleep_range(API_PAUSE_SECONDS)

    records = refresh_download_flags(records)
    save_all_records_jsonl(records, JSONL_PATH)
    write_csv_from_records(records, CSV_PATH)

    if not pending_categories:
        state["metadata_complete"] = True
        save_state(state)

    print("\nMetadata collection complete.")
    print(f"Total records: {len(records)}")
    print(f"New records this run: {new_records}")
    print(f"Remaining metadata categories: {len(pending_categories)}")
    print(f"JSONL: {JSONL_PATH}")
    print(f"CSV:   {CSV_PATH}")
    print(f"STATE: {STATE_PATH}")
    print(f"SUBCATEGORIES TXT: {SUBCATEGORIES_TXT_PATH}")


def download_image(url: str, dest: Path):
    if dest.exists():
        return "already_exists"

    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(MAX_RETRIES):
        sleep_range(DOWNLOAD_PAUSE_SECONDS)
        response = session.get(url, stream=True, timeout=REQUEST_TIMEOUT)

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait = int(retry_after)
            else:
                wait = min(60 * (2**attempt), 1800)
            print(f"429 on file download. Sleeping {wait}s before retry...")
            response.close()
            time.sleep(wait)
            continue

        try:
            response.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return "downloaded"
        finally:
            response.close()

    raise RuntimeError(f"Too many retries for file download: {url}")


def stage2_download_images(limit: int | None = None, retry_failed: bool = True):
    records, _, _ = load_existing_records(JSONL_PATH)
    if not records:
        raise FileNotFoundError(f"No metadata records found in: {JSONL_PATH}")

    state = load_state()
    if not state.get("category_discovery_complete"):
        raise RuntimeError("Category discovery is not complete.")
    if not SUBCATEGORIES_TXT_PATH.exists():
        raise RuntimeError(f"Subcategory list file missing: {SUBCATEGORIES_TXT_PATH}")

    records = refresh_download_flags(records)

    pending_statuses = {"pending"}
    if retry_failed:
        pending_statuses.add("failed")

    candidates = [
        rec
        for rec in records
        if not rec.get("is_downloaded", False)
        and rec.get("download_status", "pending") in pending_statuses
    ]

    if limit is not None:
        candidates = candidates[:limit]

    downloaded = 0
    skipped = 0
    failed = 0

    with open(FAILED_DOWNLOADS_PATH, "a", encoding="utf-8") as failed_out:
        for idx, rec in enumerate(candidates, start=1):
            image_url = rec["image_url"]
            local_path = Path(rec["local_image_path"])
            title = rec.get("title", "")

            rec["download_attempts"] = rec.get("download_attempts", 0) + 1

            try:
                status = download_image(image_url, local_path)

                if status == "already_exists":
                    rec["is_downloaded"] = True
                    rec["download_status"] = "downloaded"
                    rec["last_error"] = ""
                    skipped += 1
                else:
                    rec["is_downloaded"] = True
                    rec["download_status"] = "downloaded"
                    rec["last_error"] = ""
                    downloaded += 1

                print(f"[{idx}] {status.upper()} -> {local_path.name}")

            except Exception as e:
                rec["is_downloaded"] = False
                rec["download_status"] = "failed"
                rec["last_error"] = str(e)
                failed += 1

                error_record = {
                    "title": title,
                    "image_url": image_url,
                    "local_image_path": str(local_path),
                    "error": str(e),
                }
                failed_out.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                failed_out.flush()
                print(f"[{idx}] FAILED {title} -> {e}")

            save_all_records_jsonl(records, JSONL_PATH)
            write_csv_from_records(records, CSV_PATH)

    print("\nDownload complete.")
    print(f"Downloaded new: {downloaded}")
    print(f"Already existed: {skipped}")
    print(f"Failed:         {failed}")
    print(f"Images dir:     {IMAGES_DIR}")
    print(f"Metadata JSONL: {JSONL_PATH}")
    print(f"Metadata CSV:   {CSV_PATH}")
    print(f"Subcat txt:     {SUBCATEGORIES_TXT_PATH}")


if __name__ == "__main__":
    stage0_discover_categories(ROOT_CATEGORY, resume=True)
    stage1_collect_metadata_from_categories(
        ROOT_CATEGORY, max_categories=None, resume=True
    )
    stage2_download_images(limit=20000, retry_failed=True)
