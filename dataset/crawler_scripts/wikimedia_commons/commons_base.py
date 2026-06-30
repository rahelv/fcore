#!/usr/bin/env python3
"""
Base class for Wikimedia Commons crawlers.

Provides all shared HTTP, API, metadata extraction, record I/O, and the
3-stage category pipeline (BFS discovery → metadata collection → download).
Subclasses override _local_image_path() to change the image folder layout,
and implement run() as their entry point.
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


class CommonsCrawler:

    # ── Class-level config (override in subclasses if needed) ──────────────────

    API          = "https://commons.wikimedia.org/w/api.php"
    WIKIDATA_API = "https://www.wikidata.org/w/api.php"
    USER_AGENT   = "CostumeRecognitionBot/0.1 (contact: rha.kempf@gmail.com)"

    CATEGORY_BATCH_SIZE       = 500
    FILEINFO_BATCH_SIZE       = 25
    MEDIAINFO_BATCH_SIZE      = 50
    WIKIDATA_LABEL_BATCH_SIZE = 50

    API_PAUSE_SECONDS      = (1.5, 3.0)
    DOWNLOAD_PAUSE_SECONDS = (2.5, 5.0)
    MAX_RETRIES            = 6
    REQUEST_TIMEOUT        = 120
    MAXLAG                 = 5

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})

        # Per-category paths — set by configure_category_paths() before each root
        self.ROOT_CATEGORY          = None
        self.CATEGORY_OUT_DIR       = None
        self.METADATA_DIR           = None
        self.IMAGES_DIR             = None
        self.JSONL_PATH             = None
        self.CSV_PATH               = None
        self.FAILED_DOWNLOADS_PATH  = None
        self.STATE_PATH             = None
        self.SUBCATEGORIES_TXT_PATH = None

    # ── Utilities ──────────────────────────────────────────────────────────────

    def sleep_range(self, bounds):
        time.sleep(random.uniform(bounds[0], bounds[1]))

    def clean_html_text(self, value: str) -> str:
        if not value:
            return ""
        value = html.unescape(value)
        value = re.sub(r"<[^>]+>", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    def normalize_category_title(self, title: str) -> str:
        title = title.strip().replace("_", " ")
        if title.startswith("https://commons.wikimedia.org/wiki/"):
            title = title.rsplit("/", 1)[-1]
            title = unquote(title).replace("_", " ")
        if not title.startswith("Category:"):
            title = "Category:" + title
        return title

    def safe_filename(self, name: str) -> str:
        name = unquote(name)
        name = re.sub(r'[\\/*?:"<>|]', "_", name)
        return name.strip()

    def filename_from_url(self, url: str) -> str:
        path = urlparse(url).path
        return self.safe_filename(path.split("/")[-1])

    def category_slug(self, category_title: str) -> str:
        title = self.normalize_category_title(category_title)
        title = title.removeprefix("Category:")
        title = self.safe_filename(title)
        title = re.sub(r"\s+", "_", title)
        return title.strip("_") or "category"

    # ── Path configuration ─────────────────────────────────────────────────────

    def configure_category_paths(self, root_category: str) -> str:
        self.ROOT_CATEGORY = self.normalize_category_title(root_category)

        slug = self.category_slug(self.ROOT_CATEGORY)
        self.CATEGORY_OUT_DIR = self.out_dir / slug
        self.METADATA_DIR     = self.CATEGORY_OUT_DIR / "metadata"
        self.IMAGES_DIR       = self.CATEGORY_OUT_DIR / "images"

        self.METADATA_DIR.mkdir(parents=True, exist_ok=True)
        self.IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        self.JSONL_PATH             = self.METADATA_DIR / "metadata.jsonl"
        self.CSV_PATH               = self.METADATA_DIR / "metadata.csv"
        self.FAILED_DOWNLOADS_PATH  = self.METADATA_DIR / "failed_downloads.jsonl"
        self.STATE_PATH             = self.METADATA_DIR / "crawl_state.json"
        self.SUBCATEGORIES_TXT_PATH = self.METADATA_DIR / "all_subcategories.txt"

        return self.ROOT_CATEGORY

    # ── State ──────────────────────────────────────────────────────────────────

    def load_state(self) -> dict:
        if not self.STATE_PATH.exists():
            return {
                "root_category":                 self.ROOT_CATEGORY,
                "category_discovery_complete":   False,
                "metadata_complete":             False,
                "pending_categories_to_visit":   [self.ROOT_CATEGORY],
                "visited_categories":            [],
                "discovered_categories":         [self.ROOT_CATEGORY],
                "pending_metadata_categories":   [],
                "processed_metadata_categories": [],
            }
        with open(self.STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_state(self, state: dict):
        with open(self.STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    # ── HTTP / API ─────────────────────────────────────────────────────────────

    def request_with_retry(self, url: str, *, params=None, data=None,
                           stream: bool = False, timeout: int = None,
                           max_retries: int = None):
        timeout     = timeout     if timeout     is not None else self.REQUEST_TIMEOUT
        max_retries = max_retries if max_retries is not None else self.MAX_RETRIES

        for attempt in range(max_retries):
            if data is not None:
                response = self.session.post(url, params=params, data=data,
                                             stream=stream, timeout=timeout)
            else:
                response = self.session.get(url, params=params,
                                            stream=stream, timeout=timeout)

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait = int(retry_after) if (retry_after and retry_after.isdigit()) \
                       else min(60 * (2 ** attempt), 1800)
                print(f"429 Too Many Requests. Sleeping {wait}s and retrying...")
                response.close()
                time.sleep(wait)
                continue

            if response.status_code == 503 and not stream:
                try:
                    err = response.json().get("error", {})
                    if err.get("code") == "maxlag":
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

    def api_get(self, params: dict) -> dict:
        merged = {"format": "json", "maxlag": self.MAXLAG}
        merged.update(params)
        response = self.request_with_retry(self.API, params=merged, stream=False)
        data = response.json()
        if data.get("error", {}).get("code") == "maxlag":
            raise RuntimeError(f"MediaWiki maxlag response: {data['error']}")
        return data

    def api_post(self, params: dict) -> dict:
        merged = {"format": "json", "maxlag": self.MAXLAG}
        merged.update(params)
        response = self.request_with_retry(self.API, data=merged, stream=False)
        data = response.json()
        if data.get("error", {}).get("code") == "maxlag":
            raise RuntimeError(f"MediaWiki maxlag response: {data['error']}")
        return data

    def wikidata_post(self, params: dict) -> dict:
        merged = {"format": "json"}
        merged.update(params)
        response = self.request_with_retry(self.WIKIDATA_API, data=merged, stream=False)
        return response.json()

    # ── Commons API helpers ────────────────────────────────────────────────────

    def get_category_members(self, category_title: str, cmtype: str):
        """Yields categorymembers rows with pagination. cmtype: 'subcat' or 'file'."""
        cmcontinue = None
        while True:
            params = {
                "action":  "query",
                "list":    "categorymembers",
                "cmtitle": category_title,
                "cmtype":  cmtype,
                "cmlimit": self.CATEGORY_BATCH_SIZE,
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue
            data = self.api_get(params)
            for member in data.get("query", {}).get("categorymembers", []):
                yield member
            cmcontinue = data.get("continue", {}).get("cmcontinue")
            if not cmcontinue:
                break
            self.sleep_range(self.API_PAUSE_SECONDS)

    def get_file_pages_info(self, file_titles) -> list:
        if not file_titles:
            return []
        all_pages = []
        for start in range(0, len(file_titles), self.FILEINFO_BATCH_SIZE):
            batch = file_titles[start: start + self.FILEINFO_BATCH_SIZE]
            data  = self.api_get({
                "action": "query",
                "titles": "|".join(batch),
                "prop":   "info|categories|imageinfo",
                "inprop": "url",
                "cllimit": "max",
                "iiprop": "url|extmetadata",
            })
            all_pages.extend(data.get("query", {}).get("pages", {}).values())
            self.sleep_range(self.API_PAUSE_SECONDS)
        return all_pages

    def get_mediainfo_entities(self, m_ids) -> dict:
        if not m_ids:
            return {}
        all_entities = {}
        for start in range(0, len(m_ids), self.MEDIAINFO_BATCH_SIZE):
            batch = m_ids[start: start + self.MEDIAINFO_BATCH_SIZE]
            data  = self.api_post({
                "action": "wbgetentities",
                "ids":    "|".join(batch),
                "props":  "labels|descriptions|claims",
            })
            all_entities.update(data.get("entities", {}))
            self.sleep_range(self.API_PAUSE_SECONDS)
        return all_entities

    def get_wikidata_labels(self, qids, lang: str = "en") -> dict:
        if not qids:
            return {}
        result      = {}
        unique_qids = sorted(set(qids))
        for start in range(0, len(unique_qids), self.WIKIDATA_LABEL_BATCH_SIZE):
            batch    = unique_qids[start: start + self.WIKIDATA_LABEL_BATCH_SIZE]
            entities = self.wikidata_post({
                "action":    "wbgetentities",
                "ids":       "|".join(batch),
                "languages": lang,
                "props":     "labels",
            }).get("entities", {})
            for qid, ent in entities.items():
                result[qid] = ent.get("labels", {}).get(lang, {}).get("value", "")
            self.sleep_range(self.API_PAUSE_SECONDS)
        return result

    # ── Metadata extraction ────────────────────────────────────────────────────

    def extract_extmetadata(self, extmeta) -> dict:
        def get(name):
            return self.clean_html_text((extmeta.get(name) or {}).get("value", ""))
        return {
            "license_short":     get("LicenseShortName"),
            "license_url":       get("LicenseUrl"),
            "artist":            get("Artist"),
            "credit":            get("Credit"),
            "usage_terms":       get("UsageTerms"),
            "image_description": get("ImageDescription"),
            "object_name":       get("ObjectName"),
        }

    def extract_categories(self, page) -> list:
        cats = []
        for c in page.get("categories", []):
            title = c.get("title", "")
            if title.startswith("Category:"):
                title = title[len("Category:"):]
            cats.append(title)
        return cats

    def extract_p180_qids(self, entity) -> list:
        out = []
        for stmt in entity.get("claims", {}).get("P180", []):
            value = stmt.get("mainsnak", {}).get("datavalue", {}).get("value", {})
            if isinstance(value, dict) and "id" in value:
                out.append(value["id"])
        return out

    def build_costume_hint(self, depicts_labels, categories) -> str:
        if depicts_labels:
            return "; ".join(depicts_labels)
        costumeish = [
            c for c in categories
            if any(k in c.lower() for k in
                   ["cosplay", "costume", "character", "anime", "manga",
                    "fiction", "comic", "game"])
        ]
        return "; ".join(costumeish[:10])

    # ── Record I/O ─────────────────────────────────────────────────────────────

    def load_existing_records(self, path: Path = None):
        path = path or self.JSONL_PATH
        records = []
        by_pageid    = {}
        seen_pageids = set()
        if not path.exists():
            return records, by_pageid, seen_pageids
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec    = json.loads(line)
                records.append(rec)
                pageid = rec.get("pageid")
                if pageid is not None:
                    by_pageid[pageid] = rec
                    seen_pageids.add(pageid)
        return records, by_pageid, seen_pageids

    def save_all_records_jsonl(self, records, path: Path = None):
        path = path or self.JSONL_PATH
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def write_csv_from_records(self, records, path: Path = None):
        path = path or self.CSV_PATH
        fieldnames = [
            "pageid", "mediainfo_id", "title", "file_page_url", "image_url",
            "local_image_path", "is_downloaded", "download_status",
            "download_attempts", "last_error", "costume_hint",
            "depicts_qids", "depicts_labels", "categories",
            "source_category", "root_category",
            "image_description", "object_name", "artist", "credit",
            "license_short", "license_url", "usage_terms",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in records:
                row = row.copy()
                row["depicts_qids"]   = "|".join(row.get("depicts_qids", []))
                row["depicts_labels"] = "|".join(row.get("depicts_labels", []))
                row["categories"]     = "|".join(row.get("categories", []))
                writer.writerow(row)

    def refresh_download_flags(self, records):
        for rec in records:
            local_path = Path(rec["local_image_path"])
            exists     = local_path.exists()
            rec["is_downloaded"] = exists
            if exists:
                rec["download_status"] = "downloaded"
                rec["last_error"]      = ""
        return records

    def write_subcategories_txt(self, categories):
        with open(self.SUBCATEGORIES_TXT_PATH, "w", encoding="utf-8") as f:
            for cat in sorted(set(categories)):
                f.write(cat + "\n")

    # ── Image download ─────────────────────────────────────────────────────────

    def download_image(self, url: str, dest: Path) -> str:
        if dest.exists():
            return "already_exists"
        dest.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(self.MAX_RETRIES):
            self.sleep_range(self.DOWNLOAD_PAUSE_SECONDS)
            response = self.session.get(url, stream=True, timeout=self.REQUEST_TIMEOUT)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait = int(retry_after) if (retry_after and retry_after.isdigit()) \
                       else min(60 * (2 ** attempt), 1800)
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

    # ── Pipeline stages ────────────────────────────────────────────────────────

    def _local_image_path(self, source_category_title: str, filename: str) -> Path:
        """Returns the local path for an image. Override to change the folder layout."""
        return self.IMAGES_DIR / filename

    def stage0_discover_categories(self, root_category: str, resume: bool = True):
        root_category = self.normalize_category_title(root_category)
        state = self.load_state()

        if not resume or state.get("root_category") != root_category:
            state = {
                "root_category":                 root_category,
                "category_discovery_complete":   False,
                "metadata_complete":             False,
                "pending_categories_to_visit":   [root_category],
                "visited_categories":            [],
                "discovered_categories":         [root_category],
                "pending_metadata_categories":   [],
                "processed_metadata_categories": [],
            }

        visited    = set(state.get("visited_categories", []))
        discovered = set(state.get("discovered_categories", [root_category]))
        pending    = list(state.get("pending_categories_to_visit", [root_category]))

        print(f"Starting category discovery from root: {root_category}")

        while pending:
            current = pending.pop(0)
            if current in visited:
                continue

            print(f"[CATEGORY] {current}")
            new_subcats = 0

            for member in self.get_category_members(current, "subcat"):
                subcat = self.normalize_category_title(member.get("title", ""))
                if not subcat:
                    continue
                if subcat not in discovered:
                    discovered.add(subcat)
                    pending.append(subcat)
                    new_subcats += 1

            print(f"    new subcategories found: {new_subcats}")
            visited.add(current)

            state["pending_categories_to_visit"] = pending
            state["visited_categories"]          = sorted(visited)
            state["discovered_categories"]       = sorted(discovered)
            self.save_state(state)
            self.sleep_range(self.API_PAUSE_SECONDS)

        state["category_discovery_complete"] = True
        state["pending_categories_to_visit"] = []
        state["visited_categories"]          = sorted(visited)
        state["discovered_categories"]       = sorted(discovered)
        state["pending_metadata_categories"] = sorted(discovered)
        self.save_state(state)
        self.write_subcategories_txt(sorted(discovered))

        print("\nCategory discovery complete.")
        print(f"Total categories found, including root: {len(discovered)}")
        print(f"Subcategory list written to: {self.SUBCATEGORIES_TXT_PATH}")

    def stage1_collect_metadata_from_categories(self, root_category: str,
                                                 max_categories: int | None = None,
                                                 resume: bool = True):
        root_category = self.normalize_category_title(root_category)
        state = self.load_state()

        if not state.get("category_discovery_complete"):
            raise RuntimeError(
                "Category discovery not complete. Run stage0_discover_categories() first."
            )

        records, by_pageid, seen_pageids = self.load_existing_records()
        records = self.refresh_download_flags(records)

        if not resume:
            state["pending_metadata_categories"]   = state.get("discovered_categories", [])
            state["processed_metadata_categories"] = []

        pending_categories   = list(state.get("pending_metadata_categories", []))
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

            file_titles = [
                m.get("title", "") for m in self.get_category_members(category_title, "file")
                if m.get("title", "").startswith("File:")
            ]
            print(f"    files in category: {len(file_titles)}")

            pages        = self.get_file_pages_info(file_titles)
            page_records = []
            for page in pages:
                pageid = page.get("pageid")
                if not pageid or pageid in seen_pageids:
                    continue
                page_records.append((pageid, f"M{pageid}", page))

            print(f"    new files after dedupe: {len(page_records)}")

            mediainfo = self.get_mediainfo_entities([mid for _, mid, _ in page_records])
            self.sleep_range(self.API_PAUSE_SECONDS)

            all_qids, p180_by_mid = [], {}
            for _, mid, _ in page_records:
                qids = self.extract_p180_qids(mediainfo.get(mid, {}))
                p180_by_mid[mid] = qids
                all_qids.extend(qids)

            qid_to_label = self.get_wikidata_labels(all_qids)
            self.sleep_range(self.API_PAUSE_SECONDS)

            for pageid, mid, page in page_records:
                title     = page.get("title", "")
                fullurl   = page.get("fullurl", "")
                imageinfo = (page.get("imageinfo") or [{}])[0]
                image_url = imageinfo.get("url", "")

                if not image_url:
                    print(f"SKIP {title} -> no image URL")
                    continue

                ext            = self.extract_extmetadata(imageinfo.get("extmetadata", {}))
                categories     = self.extract_categories(page)
                depicts_qids   = p180_by_mid.get(mid, [])
                depicts_labels = [qid_to_label.get(qid, qid) for qid in depicts_qids]
                hint           = self.build_costume_hint(depicts_labels, categories)
                filename       = self.filename_from_url(image_url)
                local_path     = self._local_image_path(category_title, filename)

                record = {
                    "pageid":            pageid,
                    "mediainfo_id":      mid,
                    "title":             title,
                    "file_page_url":     fullurl,
                    "image_url":         image_url,
                    "local_image_path":  str(local_path),
                    "is_downloaded":     local_path.exists(),
                    "download_status":   "downloaded" if local_path.exists() else "pending",
                    "download_attempts": 0,
                    "last_error":        "",
                    "costume_hint":      hint,
                    "depicts_qids":      depicts_qids,
                    "depicts_labels":    depicts_labels,
                    "categories":        categories,
                    "source_category":   category_title,
                    "root_category":     root_category,
                    **ext,
                }
                records.append(record)
                by_pageid[pageid] = record
                seen_pageids.add(pageid)
                new_records += 1

                print(
                    f"[NEW {new_records}] METADATA -> {title}\n"
                    f"    source_category: {category_title}\n"
                    f"    downloaded: {local_path.exists()}\n"
                    f"    costume_hint: {hint or '(none)'}"
                )

            processed_categories.add(category_title)
            categories_done_this_run += 1

            self.save_all_records_jsonl(records)
            self.write_csv_from_records(records)

            state["pending_metadata_categories"]   = pending_categories
            state["processed_metadata_categories"] = sorted(processed_categories)
            state["metadata_complete"]             = len(pending_categories) == 0
            self.save_state(state)
            self.sleep_range(self.API_PAUSE_SECONDS)

        records = self.refresh_download_flags(records)
        self.save_all_records_jsonl(records)
        self.write_csv_from_records(records)

        if not pending_categories:
            state["metadata_complete"] = True
            self.save_state(state)

        print("\nMetadata collection complete.")
        print(f"Total records: {len(records)}")
        print(f"New records this run: {new_records}")
        print(f"Remaining metadata categories: {len(pending_categories)}")
        print(f"JSONL: {self.JSONL_PATH}")
        print(f"CSV:   {self.CSV_PATH}")
        print(f"STATE: {self.STATE_PATH}")
        print(f"SUBCATEGORIES TXT: {self.SUBCATEGORIES_TXT_PATH}")

    def stage2_download_images(self, limit: int | None = None, retry_failed: bool = True):
        records, _, _ = self.load_existing_records()
        if not records:
            raise FileNotFoundError(f"No metadata records found in: {self.JSONL_PATH}")

        state = self.load_state()
        if not state.get("category_discovery_complete"):
            raise RuntimeError("Category discovery is not complete.")
        if not self.SUBCATEGORIES_TXT_PATH.exists():
            raise RuntimeError(f"Subcategory list file missing: {self.SUBCATEGORIES_TXT_PATH}")

        records = self.refresh_download_flags(records)

        pending_statuses = {"pending"}
        if retry_failed:
            pending_statuses.add("failed")

        candidates = [
            rec for rec in records
            if not rec.get("is_downloaded", False)
            and rec.get("download_status", "pending") in pending_statuses
        ]
        if limit is not None:
            candidates = candidates[:limit]

        downloaded = skipped = failed = 0

        with open(self.FAILED_DOWNLOADS_PATH, "a", encoding="utf-8") as failed_out:
            for idx, rec in enumerate(candidates, start=1):
                image_url  = rec["image_url"]
                local_path = Path(rec["local_image_path"])
                title      = rec.get("title", "")

                rec["download_attempts"] = rec.get("download_attempts", 0) + 1

                try:
                    status = self.download_image(image_url, local_path)
                    rec["is_downloaded"]   = True
                    rec["download_status"] = "downloaded"
                    rec["last_error"]      = ""
                    if status == "already_exists":
                        skipped += 1
                    else:
                        downloaded += 1
                    print(f"[{idx}] {status.upper()} -> {local_path.name}")
                except Exception as e:
                    rec["is_downloaded"]   = False
                    rec["download_status"] = "failed"
                    rec["last_error"]      = str(e)
                    failed += 1
                    failed_out.write(json.dumps({
                        "title": title, "image_url": image_url,
                        "local_image_path": str(local_path), "error": str(e),
                    }, ensure_ascii=False) + "\n")
                    failed_out.flush()
                    print(f"[{idx}] FAILED {title} -> {e}")

                self.save_all_records_jsonl(records)
                self.write_csv_from_records(records)

        print("\nDownload complete.")
        print(f"Downloaded new: {downloaded}")
        print(f"Already existed: {skipped}")
        print(f"Failed:         {failed}")
        print(f"Images dir:     {self.IMAGES_DIR}")
        print(f"Metadata JSONL: {self.JSONL_PATH}")
        print(f"Metadata CSV:   {self.CSV_PATH}")
        print(f"Subcat txt:     {self.SUBCATEGORIES_TXT_PATH}")

    def process_root_category(self, root_category: str, *, resume: bool = True,
                               max_categories: int | None = None,
                               download_limit: int | None = None,
                               retry_failed: bool = True):
        root_category = self.configure_category_paths(root_category)

        print("\n" + "=" * 80)
        print(f"PROCESSING ROOT CATEGORY: {root_category}")
        print(f"OUTPUT DIR: {self.CATEGORY_OUT_DIR}")
        print("=" * 80 + "\n")

        self.stage0_discover_categories(root_category, resume=resume)
        self.stage1_collect_metadata_from_categories(
            root_category, max_categories=max_categories, resume=resume
        )
        self.stage2_download_images(limit=download_limit, retry_failed=retry_failed)

    # ── Entry point ────────────────────────────────────────────────────────────

    def run(self):
        raise NotImplementedError("Subclasses must implement run()")
