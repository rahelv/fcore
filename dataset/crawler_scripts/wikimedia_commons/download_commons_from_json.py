#!/usr/bin/env python3
"""
Download Wikimedia Commons images from all categories listed in a JSON file.

Input JSON format:
[
  {
    "category": "Category:Cosplay of Ariel",
    "name": "Cosplay of Ariel",
    "url": "https://commons.wikimedia.org/wiki/Category:Cosplay_of_Ariel",
    "keep": "",
    "probably_useless": false
  },
  ...
]

What this script does:
1. Reads all categories from the JSON.
2. For each JSON category, discovers all nested subcategories.
3. Collects file metadata from the root category and all discovered subcategories.
4. Downloads images into separate folders per source category:

   OUT_DIR/
     Cosplay_of_Ariel/
       metadata/
         metadata.jsonl
         metadata.csv
         crawl_state.json
         all_subcategories.txt
       images/
         Cosplay_of_Ariel/
           image1.jpg
         Cosplay_of_Ariel_in_Belgium/
           image2.jpg

This keeps subcategory images separated instead of mixing all images into one folder.
"""

import argparse
import json
from pathlib import Path

from commons_base import CommonsCrawler


class JsonCategoryCrawler(CommonsCrawler):
    """Downloads Wikimedia Commons images from categories listed in a JSON file."""

    DEFAULT_OUT_DIR = Path("/home/ubuntu/data/data/disney_characters_commons")

    def __init__(self, out_dir: Path | None = None):
        super().__init__(out_dir or self.DEFAULT_OUT_DIR)

    # Per-source subfolder layout: OUT_DIR/<root_slug>/images/<source_slug>/<filename>
    def _local_image_path(self, source_category_title: str, filename: str) -> Path:
        return self.IMAGES_DIR / self.category_slug(source_category_title) / filename

    def load_categories_from_json(self, json_path: Path, only_keep: bool = False,
                                   skip_probably_useless: bool = False) -> list[str]:
        """
        Reads category titles from a JSON file.
        Uses the "category" field first, then falls back to "url", then "name".
        Dedupes by normalized category title while preserving order.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            rows = json.load(f)

        if not isinstance(rows, list):
            raise ValueError("Expected JSON file to contain a list of category objects.")

        categories = []
        seen = set()

        for row in rows:
            if not isinstance(row, dict):
                continue
            if only_keep and str(row.get("keep", "")).lower() not in {"yes", "y", "true", "1"}:
                continue
            if skip_probably_useless and bool(row.get("probably_useless", False)):
                continue
            raw = row.get("category") or row.get("url") or row.get("name")
            if not raw:
                continue
            cat = self.normalize_category_title(raw)
            if cat not in seen:
                seen.add(cat)
                categories.append(cat)

        return categories

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--categories-json",
            required=True,
            help="JSON file containing category entries with a 'category', 'url', or 'name' field.",
        )
        parser.add_argument(
            "--out-dir",
            default=str(self.DEFAULT_OUT_DIR),
            help="Base output directory.",
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            default=True,
            help="Resume from crawl_state.json if present. Default: true.",
        )
        parser.add_argument(
            "--no-resume",
            action="store_true",
            help="Start fresh for each category.",
        )
        parser.add_argument(
            "--only-keep",
            action="store_true",
            help="Only process JSON rows where keep is yes/y/true/1.",
        )
        parser.add_argument(
            "--skip-probably-useless",
            action="store_true",
            help="Skip rows where probably_useless is true.",
        )
        parser.add_argument(
            "--max-roots",
            type=int,
            default=None,
            help="Process only the first N root categories from the JSON. Useful for testing.",
        )
        parser.add_argument(
            "--max-categories-per-root",
            type=int,
            default=None,
            help="Limit metadata category processing per root. Useful for testing.",
        )
        parser.add_argument(
            "--download-limit-per-root",
            type=int,
            default=None,
            help="Limit number of image downloads per root. Useful for testing.",
        )
        parser.add_argument(
            "--no-retry-failed",
            action="store_true",
            help="Do not retry previously failed downloads.",
        )
        return parser.parse_args()

    def run(self):
        args = self.parse_args()

        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        categories = self.load_categories_from_json(
            Path(args.categories_json),
            only_keep=args.only_keep,
            skip_probably_useless=args.skip_probably_useless,
        )
        if args.max_roots is not None:
            categories = categories[: args.max_roots]

        print(f"Loaded {len(categories)} root categories from {args.categories_json}")

        for category in categories:
            self.process_root_category(
                category,
                resume=not args.no_resume,
                max_categories=args.max_categories_per_root,
                download_limit=args.download_limit_per_root,
                retry_failed=not args.no_retry_failed,
            )


if __name__ == "__main__":
    JsonCategoryCrawler().run()
