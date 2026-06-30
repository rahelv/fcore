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

import json
from pathlib import Path

from commons_base import CommonsCrawler


class CharacterPipelineCrawler(CommonsCrawler):
    """Crawls Wikimedia Commons images organised by cosplay character."""

    CHARACTER_ROOT = "Category:Cosplay by character"
    MIN_IMAGES     = 50
    OUT_DIR        = Path("data/characters")

    def __init__(self):
        super().__init__(self.OUT_DIR)
        self.PIPELINE_STATE_PATH  = self.out_dir / "pipeline_state.json"
        self.QUALIFYING_CHARS_PATH = self.out_dir / "qualifying_characters.json"

    # ── Character-specific utilities ───────────────────────────────────────────

    def character_slug(self, category_title: str) -> str:
        """'Category:Cosplay of Hu Tao' → 'hu_tao'"""
        import re
        title = self.normalize_category_title(category_title)
        name  = title.removeprefix("Category:Cosplay of ").strip()
        name  = re.sub(r"\s+", "_", name)
        name  = re.sub(r'[\\/*?:"<>|]', "", name)
        return name.lower().strip("_") or "unknown"

    def character_name_from_category(self, category_title: str) -> str:
        """'Category:Cosplay of Hu Tao' → 'Hu Tao'"""
        title = self.normalize_category_title(category_title)
        return title.removeprefix("Category:Cosplay of ").strip()

    # ── Recursive image counting ───────────────────────────────────────────────

    def get_category_direct_file_count(self, category_title: str) -> int:
        """Returns the number of files directly in a category (non-recursive)."""
        data = self.api_get({
            "action": "query",
            "prop":   "categoryinfo",
            "titles": category_title,
        })
        for page in data.get("query", {}).get("pages", {}).values():
            return page.get("categoryinfo", {}).get("files", 0)
        return 0

    def count_images_recursive(self, category_title: str) -> int:
        """
        BFS over the category tree starting at category_title.
        Uses categoryinfo (one call per category) to count direct files,
        avoiding the need to list every file individually.
        """
        visited: set[str] = set()
        pending = [self.normalize_category_title(category_title)]
        total   = 0

        while pending:
            current = pending.pop(0)
            if current in visited:
                continue
            visited.add(current)

            total += self.get_category_direct_file_count(current)
            self.sleep_range(self.API_PAUSE_SECONDS)

            for member in self.get_category_members(current, "subcat"):
                subcat = self.normalize_category_title(member.get("title", ""))
                if subcat and subcat not in visited:
                    pending.append(subcat)
            self.sleep_range(self.API_PAUSE_SECONDS)

        return total

    # ── Pipeline state ─────────────────────────────────────────────────────────

    def load_pipeline_state(self) -> dict:
        if self.PIPELINE_STATE_PATH.exists():
            with open(self.PIPELINE_STATE_PATH, encoding="utf-8") as f:
                return json.load(f)
        return {
            "stage0_complete":        False,
            "character_categories":   [],
            "stage1_complete":        False,
            "image_counts":           {},
            "qualifying_characters":  [],
            "downloaded_characters":  [],
        }

    def save_pipeline_state(self, state: dict):
        self.PIPELINE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.PIPELINE_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    # ── Stage 0: discover character categories ─────────────────────────────────

    def stage0_discover_character_categories(self, state: dict) -> dict:
        if state.get("stage0_complete"):
            print(f"Stage 0 already complete ({len(state['character_categories'])} character categories).")
            return state

        print(f"\n{'='*70}")
        print(f"Stage 0: Discovering character categories under {self.CHARACTER_ROOT}")
        print(f"{'='*70}")

        found = set(state.get("character_categories", []))

        for member in self.get_category_members(self.CHARACTER_ROOT, "subcat"):
            title = self.normalize_category_title(member.get("title", ""))
            if title and title not in found:
                found.add(title)
                print(f"  Found: {title}")
            self.sleep_range((0.1, 0.3))

        state["character_categories"] = sorted(found)
        state["stage0_complete"]      = True
        self.save_pipeline_state(state)

        print(f"\nStage 0 complete. Found {len(found)} character categories.")
        return state

    # ── Stage 1: count images recursively per character ────────────────────────

    def stage1_count_and_filter(self, state: dict) -> dict:
        if state.get("stage1_complete"):
            q = state.get("qualifying_characters", [])
            print(f"Stage 1 already complete ({len(q)} qualifying characters).")
            return state

        print(f"\n{'='*70}")
        print(f"Stage 1: Counting images per character (threshold >= {self.MIN_IMAGES})")
        print(f"{'='*70}")

        character_categories = [
            c for c in state["character_categories"]
            if self.normalize_category_title(c).startswith("Category:Cosplay of ")
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
            count = self.count_images_recursive(cat)
            image_counts[cat] = count
            print(f"{count} images")

            state["image_counts"] = image_counts
            self.save_pipeline_state(state)

        qualifying = sorted(
            cat for cat, cnt in image_counts.items() if cnt >= self.MIN_IMAGES
        )

        state["image_counts"]           = image_counts
        state["qualifying_characters"]  = qualifying
        state["stage1_complete"]        = True
        self.save_pipeline_state(state)

        with open(self.QUALIFYING_CHARS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "category":       cat,
                        "character_name": self.character_name_from_category(cat),
                        "image_count":    image_counts[cat],
                        "slug":           self.character_slug(cat),
                    }
                    for cat in qualifying
                ],
                f, ensure_ascii=False, indent=2,
            )

        print(f"\nStage 1 complete. {len(qualifying)} characters qualify (>= {self.MIN_IMAGES} images).")
        print(f"Qualifying list written to: {self.QUALIFYING_CHARS_PATH}")
        return state

    # ── Stage 2: download images for qualifying characters ─────────────────────

    def download_character(self, root_category: str, char_out_dir: Path):
        """
        Full pipeline for one character category:
          sub-stage A: discover all subcategories (BFS)
          sub-stage B: collect file metadata
          sub-stage C: download images
        State is saved inside char_out_dir/metadata/.
        """
        meta_dir   = char_out_dir / "metadata"
        images_dir = char_out_dir / "images"
        meta_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        state_path   = meta_dir / "crawl_state.json"
        jsonl_path   = meta_dir / "metadata.jsonl"
        csv_path     = meta_dir / "metadata.csv"
        failed_path  = meta_dir / "failed_downloads.jsonl"
        subcats_path = meta_dir / "all_subcategories.txt"

        root_category = self.normalize_category_title(root_category)

        if state_path.exists():
            with open(state_path, encoding="utf-8") as f:
                char_state = json.load(f)
        else:
            char_state = {
                "root_category":                 root_category,
                "subcat_discovery_complete":     False,
                "metadata_complete":             False,
                "pending_categories":            [root_category],
                "visited_categories":            [],
                "discovered_categories":         [root_category],
                "pending_metadata_categories":   [],
                "processed_metadata_categories": [],
            }

        def _save():
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(char_state, f, ensure_ascii=False, indent=2)

        # Sub-stage A: discover subcategories
        if not char_state.get("subcat_discovery_complete"):
            visited    = set(char_state.get("visited_categories", []))
            discovered = set(char_state.get("discovered_categories", [root_category]))
            pending    = list(char_state.get("pending_categories", [root_category]))

            while pending:
                current = pending.pop(0)
                if current in visited:
                    continue
                print(f"    [subcat] {current}")
                for member in self.get_category_members(current, "subcat"):
                    subcat = self.normalize_category_title(member.get("title", ""))
                    if subcat and subcat not in discovered:
                        discovered.add(subcat)
                        pending.append(subcat)
                visited.add(current)
                char_state.update({
                    "pending_categories":    pending,
                    "visited_categories":    sorted(visited),
                    "discovered_categories": sorted(discovered),
                })
                _save()
                self.sleep_range(self.API_PAUSE_SECONDS)

            char_state["subcat_discovery_complete"]   = True
            char_state["pending_metadata_categories"] = sorted(discovered)
            _save()

            with open(subcats_path, "w", encoding="utf-8") as f:
                for cat in sorted(discovered):
                    f.write(cat + "\n")

        # Sub-stage B: collect metadata
        if not char_state.get("metadata_complete"):
            records, by_pageid, seen_pageids = self.load_existing_records(jsonl_path)
            records = self.refresh_download_flags(records)

            pending_meta   = list(char_state.get("pending_metadata_categories", []))
            processed_meta = set(char_state.get("processed_metadata_categories", []))
            new_records    = 0

            while pending_meta:
                cat_title = pending_meta.pop(0)
                if cat_title in processed_meta:
                    continue

                file_titles = [
                    m.get("title", "") for m in self.get_category_members(cat_title, "file")
                    if m.get("title", "").startswith("File:")
                ]
                print(f"    [meta] {cat_title} — {len(file_titles)} files")

                pages        = self.get_file_pages_info(file_titles)
                page_records = [
                    (p.get("pageid"), f"M{p.get('pageid')}", p)
                    for p in pages
                    if p.get("pageid") and p["pageid"] not in seen_pageids
                ]

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
                    imageinfo = (page.get("imageinfo") or [{}])[0]
                    image_url = imageinfo.get("url", "")
                    if not image_url:
                        continue

                    ext            = self.extract_extmetadata(imageinfo.get("extmetadata", {}))
                    categories     = self.extract_categories(page)
                    depicts_qids   = p180_by_mid.get(mid, [])
                    depicts_labels = [qid_to_label.get(q, q) for q in depicts_qids]
                    local_path     = images_dir / self.filename_from_url(image_url)

                    record = {
                        "pageid":            pageid,
                        "mediainfo_id":      mid,
                        "title":             page.get("title", ""),
                        "file_page_url":     page.get("fullurl", ""),
                        "image_url":         image_url,
                        "local_image_path":  str(local_path),
                        "is_downloaded":     local_path.exists(),
                        "download_status":   "downloaded" if local_path.exists() else "pending",
                        "download_attempts": 0,
                        "last_error":        "",
                        "costume_hint":      self.build_costume_hint(depicts_labels, categories),
                        "depicts_qids":      depicts_qids,
                        "depicts_labels":    depicts_labels,
                        "categories":        categories,
                        "source_category":   cat_title,
                        "root_category":     root_category,
                        **ext,
                    }
                    records.append(record)
                    by_pageid[pageid] = record
                    seen_pageids.add(pageid)
                    new_records += 1

                processed_meta.add(cat_title)
                self.save_all_records_jsonl(records, jsonl_path)
                self.write_csv_from_records(records, csv_path)
                char_state["pending_metadata_categories"]   = pending_meta
                char_state["processed_metadata_categories"] = sorted(processed_meta)
                char_state["metadata_complete"]             = len(pending_meta) == 0
                _save()
                self.sleep_range(self.API_PAUSE_SECONDS)

            print(f"    Metadata complete. {new_records} new records.")

        # Sub-stage C: download images
        records, _, _ = self.load_existing_records(jsonl_path)
        records = self.refresh_download_flags(records)
        candidates = [
            r for r in records
            if not r.get("is_downloaded") and r.get("download_status") in ("pending", "failed")
        ]

        downloaded = skipped = failed = 0
        with open(failed_path, "a", encoding="utf-8") as fail_f:
            for idx, rec in enumerate(candidates, start=1):
                url  = rec["image_url"]
                dest = Path(rec["local_image_path"])
                rec["download_attempts"] = rec.get("download_attempts", 0) + 1
                try:
                    status = self.download_image(url, dest)
                    rec["is_downloaded"]   = True
                    rec["download_status"] = "downloaded"
                    rec["last_error"]      = ""
                    if status == "already_exists":
                        skipped += 1
                    else:
                        downloaded += 1
                    print(f"    [{idx}] {status.upper()} → {dest.name}")
                except Exception as e:
                    rec["is_downloaded"]   = False
                    rec["download_status"] = "failed"
                    rec["last_error"]      = str(e)
                    failed += 1
                    fail_f.write(json.dumps({"url": url, "error": str(e)}, ensure_ascii=False) + "\n")
                    fail_f.flush()
                    print(f"    [{idx}] FAILED {url} — {e}")

                self.save_all_records_jsonl(records, jsonl_path)
                self.write_csv_from_records(records, csv_path)

        print(f"    Downloads: {downloaded} new, {skipped} existed, {failed} failed.")

    def stage2_download_qualifying(self, state: dict) -> dict:
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

            slug     = self.character_slug(cat)
            char_dir = self.out_dir / slug / "wikimedia"
            print(f"\n[{i}/{len(qualifying)}] {cat} → {char_dir}")

            try:
                self.download_character(cat, char_dir)
                downloaded_set.add(cat)
                state["downloaded_characters"] = sorted(downloaded_set)
                self.save_pipeline_state(state)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"  ERROR for {cat}: {e}")

        print("\nStage 2 complete.")
        return state

    # ── Entry point ────────────────────────────────────────────────────────────

    def run(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        state = self.load_pipeline_state()

        state = self.stage0_discover_character_categories(state)
        state = self.stage1_count_and_filter(state)
        state = self.stage2_download_qualifying(state)

        print("\nPipeline complete.")
        print(f"Qualifying characters JSON: {self.QUALIFYING_CHARS_PATH}")
        print(f"Data directory: {self.out_dir}")


if __name__ == "__main__":
    CharacterPipelineCrawler().run()
