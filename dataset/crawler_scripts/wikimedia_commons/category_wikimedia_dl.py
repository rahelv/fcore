#!/usr/bin/env python3
from pathlib import Path

from commons_base import CommonsCrawler


class CategoryCrawler(CommonsCrawler):
    """Downloads Wikimedia Commons images from a hardcoded list of cosplay categories."""

    OUT_DIR = Path("/Users/rahel/code/FS26/playground/data/categories")

    ROOT_CATEGORIES = [
        "Category:Cosplay of Mandalorians",
        "Category:Cosplay of March 7th",
        "Category:Cosplay of Naruto Uzumaki",
        "Category:Cosplay of Raiden Shogun",
        "Category:Cosplay of Goku",
        "Category:Cosplay of Sailor Moon (character)",
        "Category:Cosplay of Sakura Haruno",
        "Category:Cosplay of Harley Quinn",
        "Category:Cosplay of Vi (League of Legends)",
        "Category:Cosplay of Hinata Hyuga",
        "Category:Cosplay of Wolverine",
        "Category:Cosplay of Izuku Midoriya",
        "Category:Cosplay of Jujutsu Kaisen",
    ]

    def __init__(self):
        super().__init__(self.OUT_DIR)

    # Flat image layout: OUT_DIR/<root_slug>/images/<filename>
    # This is the default from CommonsCrawler; no override needed.

    def run(self):
        for category in self.ROOT_CATEGORIES:
            self.process_root_category(
                category,
                resume=True,
                max_categories=None,
                download_limit=None,
                retry_failed=True,
            )


if __name__ == "__main__":
    CategoryCrawler().run()
