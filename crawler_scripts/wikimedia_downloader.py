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

SEARCH_TERM = "cosplay"
USER_AGENT = "CostumeRecognitionBot/0.1 (contact: rha.kempf@gmail.com)"

OUT_DIR = Path("/home/ubuntu/data/commons_cosplay_dataset")
METADATA_DIR = OUT_DIR / "metadata"
IMAGES_DIR = OUT_DIR / "images"
METADATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

JSONL_PATH = METADATA_DIR / "metadata.jsonl"
CSV_PATH = METADATA_DIR / "metadata.csv"
FAILED_DOWNLOADS_PATH = METADATA_DIR / "failed_downloads.jsonl"
STATE_PATH = METADATA_DIR / "crawl_state.json"

SEARCH_BATCH_SIZE = 25
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
            "search_term": SEARCH_TERM,
            "next_offset": 0,
            "metadata_complete": False,
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
    stream: bool = False,
    timeout: int = REQUEST_TIMEOUT,
    max_retries: int = MAX_RETRIES,
):
    for attempt in range(max_retries):
        response = session.get(url, params=params, stream=stream, timeout=timeout)

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait = int(retry_after)
            else:
                wait = min(60 * (2**attempt), 1800)
            print(f"429 Too Many Requests. Sleeping {wait}s and retrying...")
            time.sleep(wait)
            continue

        if response.status_code == 503 and not stream:
            try:
                data = response.json()
                error = data.get("error", {})
                if error.get("code") == "maxlag":
                    wait = min(10 * (attempt + 1), 60)
                    print(f"maxlag hit. Sleeping {wait}s and retrying...")
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


def wikidata_get(params: dict) -> dict:
    response = request_with_retry(WIKIDATA_API, params=params, stream=False)
    return response.json()


def search_files(term: str, offset: int = 0, limit: int = SEARCH_BATCH_SIZE) -> dict:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": term,
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "gsroffset": offset,
        "prop": "info|categories|imageinfo",
        "inprop": "url",
        "cllimit": "max",
        "iiprop": "url|extmetadata",
    }
    return api_get(params)


def get_mediainfo_entities(m_ids):
    if not m_ids:
        return {}
    params = {
        "action": "wbgetentities",
        "ids": "|".join(m_ids),
        "props": "labels|descriptions|claims",
        "format": "json",
        "maxlag": MAXLAG,
    }
    response = request_with_retry(API, params=params, stream=False)
    return response.json().get("entities", {})


def get_wikidata_labels(qids, lang="en"):
    if not qids:
        return {}
    params = {
        "action": "wbgetentities",
        "format": "json",
        "ids": "|".join(sorted(set(qids))),
        "languages": lang,
        "props": "labels",
    }
    entities = wikidata_get(params).get("entities", {})
    return {
        qid: ent.get("labels", {}).get(lang, {}).get("value", "")
        for qid, ent in entities.items()
    }


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
        "image_description",
        "object_name",
        "artist",
        "credit",
        "license_short",
        "license_url",
        "usage_terms",
        "search_term",
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


def stage1_collect_metadata(
    search_term: str, max_results: int | None = None, resume: bool = True
):
    records, by_pageid, seen_pageids = load_existing_records(JSONL_PATH)
    records = refresh_download_flags(records)

    state = load_state()
    if not resume or state.get("search_term") != search_term:
        state = {
            "search_term": search_term,
            "next_offset": 0,
            "metadata_complete": False,
        }

    offset = state.get("next_offset", 0)
    new_records = 0
    scanned_this_run = 0

    print(f"Starting metadata crawl from offset={offset}")

    while True:
        if max_results is not None and scanned_this_run >= max_results:
            break

        batch_limit = SEARCH_BATCH_SIZE
        if max_results is not None:
            batch_limit = min(batch_limit, max_results - scanned_this_run)
            if batch_limit <= 0:
                break

        data = search_files(search_term, offset=offset, limit=batch_limit)
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            state["metadata_complete"] = True
            save_state(state)
            break

        page_records = []
        m_ids = []

        for _, page in pages.items():
            scanned_this_run += 1
            pageid = page.get("pageid")
            if not pageid or pageid in seen_pageids:
                continue

            mid = f"M{pageid}"
            m_ids.append(mid)
            page_records.append((pageid, mid, page))

        sleep_range(API_PAUSE_SECONDS)

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
                "image_description": ext["image_description"],
                "object_name": ext["object_name"],
                "artist": ext["artist"],
                "credit": ext["credit"],
                "license_short": ext["license_short"],
                "license_url": ext["license_url"],
                "usage_terms": ext["usage_terms"],
                "search_term": search_term,
            }

            records.append(record)
            by_pageid[pageid] = record
            seen_pageids.add(pageid)
            new_records += 1

            print(
                f"[NEW {new_records}] METADATA -> {title}\n"
                f"    downloaded: {already_downloaded}\n"
                f"    costume_hint: {costume_hint or '(none)'}"
            )

        save_all_records_jsonl(records, JSONL_PATH)

        if "continue" not in data:
            state["metadata_complete"] = True
            save_state(state)
            break

        offset = data["continue"].get("gsroffset", offset + batch_limit)
        state["search_term"] = search_term
        state["next_offset"] = offset
        state["metadata_complete"] = False
        save_state(state)

        print(f"--- next search batch, offset={offset} ---")
        sleep_range(API_PAUSE_SECONDS)

    records = refresh_download_flags(records)
    save_all_records_jsonl(records, JSONL_PATH)
    write_csv_from_records(records, CSV_PATH)

    print("\nMetadata collection complete.")
    print(f"Total records: {len(records)}")
    print(f"New records this run: {new_records}")
    print(f"Next offset: {load_state().get('next_offset', 0)}")
    print(f"JSONL: {JSONL_PATH}")
    print(f"CSV:   {CSV_PATH}")
    print(f"STATE: {STATE_PATH}")


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
            time.sleep(wait)
            response.close()
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


if __name__ == "__main__":
    stage1_collect_metadata(SEARCH_TERM, max_results=None, resume=True)
    stage2_download_images(limit=20000, retry_failed=True)
