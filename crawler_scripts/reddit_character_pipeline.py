#!/usr/bin/env python3
"""
Reddit cosplay image pipeline.

Reads qualifying_characters.json produced by wikimedia_character_pipeline.py,
then for each character:
  - Searches Reddit for "<character_name> cosplay" with type=media, sort=relevance
  - Downloads up to MAX_IMAGES unique images (images only, no videos)
  - Deduplicates by URL and by content hash (catches cross-posts with different URLs)
  - Saves images to <OUT_DIR>/<slug>/reddit/images/
  - Writes per-character metadata.jsonl

Run wikimedia_character_pipeline.py first, or create your own
qualifying_characters.json with entries like:
  [{"character_name": "Hu Tao", "slug": "hu_tao", ...}, ...]
"""

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, quote_plus

import requests

# ── Config ──────────────────────────────────────────────────────────────────

QUALIFYING_CHARS_PATH = Path(
    "/home/ubuntu/data/data/characters/qualifying_characters.json"
)  # TODO: change to node path ...
OUT_DIR = Path("/home/ubuntu/data/data/characters")

MAX_IMAGES = 150  # max unique images to download per character
SORT = "relevance"
TIME_FILTER = "all"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

SLEEP_BETWEEN_DOWNLOADS = 1.0
SLEEP_BETWEEN_PAGES = 1.5
SLEEP_BETWEEN_CHARACTERS = 3.0

USER_AGENT = "CostumeRecognitionBot/0.1 (contact: rha.kempf@gmail.com)"

# ── Utilities ────────────────────────────────────────────────────────────────


def sanitize_filename(name: str, max_len: int = 180) -> str:
    name = re.sub(r"[^\w\-. ]+", "_", name, flags=re.UNICODE).strip()
    name = re.sub(r"\s+", " ", name)
    return (name or "file")[:max_len].rstrip(" .")


def is_image_url(url: str) -> bool:
    ext = Path(urlparse(url).path).suffix.lower()
    return ext in IMAGE_EXTENSIONS


# ── Reddit API ───────────────────────────────────────────────────────────────


def reddit_search(
    session: requests.Session,
    query: str,
    after: str | None = None,
    limit: int = 150,
    sort: str = "relevance",
    t: str = "all",
) -> dict:
    params = {
        "q": query,
        "sort": sort,
        "t": t,
        "limit": str(limit),
        "raw_json": "1",
        "type": "link",
    }
    if after:
        params["after"] = after

    resp = session.get(
        "https://www.reddit.com/search/.json",
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def extract_image_urls(post: dict) -> list[dict]:
    """Extract image URLs from a post, skipping videos. Returns list of {url, kind}.

    Priority (mutually exclusive per post type):
      1. Gallery  -- multi-image posts; each image is kept individually.
      2. Direct   -- post URL points straight at an image (i.redd.it, imgur, ...).
      3. Preview  -- fallback; Reddit CDN thumbnail used only when there is no
                    direct image URL. Avoids the preview+direct duplicate pair
                    that appears when both fields are populated for the same image.
    """
    # 1. Gallery posts -- may yield multiple images
    if post.get("is_gallery") and post.get("media_metadata"):
        items = []
        for item in (post.get("gallery_data") or {}).get("items", []):
            meta = post["media_metadata"].get(item.get("media_id") or "", {})
            s = meta.get("s", {})
            url = (s.get("u") or s.get("gif") or "").replace("&amp;", "&")
            if url and is_image_url(url):
                items.append({"url": url, "kind": "gallery"})
        if items:
            return items

    # 2. Direct image URL (i.redd.it, imgur, etc.) -- preferred over preview
    direct_url = post.get("url_overridden_by_dest") or post.get("url") or ""
    if direct_url and is_image_url(direct_url):
        return [{"url": direct_url, "kind": "direct"}]

    # 3. Preview fallback -- only reached when the post URL is not itself an image
    preview_images = (post.get("preview") or {}).get("images", [])
    if preview_images:
        src = preview_images[0].get("source", {})
        url = (src.get("url") or "").replace("&amp;", "&")
        if url and is_image_url(url):
            return [{"url": url, "kind": "preview"}]

    return []


def download_file(session: requests.Session, url: str, out_path: Path) -> str:
    """Download url, write to out_path, and return the SHA-256 hex digest."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with session.get(
        url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=60
    ) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=256 * 1024):
                if chunk:
                    f.write(chunk)
                    h.update(chunk)
    return h.hexdigest()


# ── Per-character download ────────────────────────────────────────────────────


def download_character_reddit(
    character_name: str, slug: str, session: requests.Session
):
    query = f"{character_name} cosplay"
    search_url = f"https://www.reddit.com/search/?q={quote_plus(query)}&type=media"

    out_dir = OUT_DIR / slug / "reddit" / "images"
    meta_path = OUT_DIR / slug / "reddit" / "metadata.jsonl"
    seen_path = OUT_DIR / slug / "reddit" / "seen_urls.json"
    hashes_path = OUT_DIR / slug / "reddit" / "seen_hashes.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already-seen URLs for resumability
    seen_urls: set[str] = set()
    if seen_path.exists():
        with open(seen_path, encoding="utf-8") as f:
            seen_urls = set(json.load(f))

    # Load already-seen content hashes (catches cross-posts with different URLs)
    seen_hashes: set[str] = set()
    if hashes_path.exists():
        with open(hashes_path, encoding="utf-8") as f:
            seen_hashes = set(json.load(f))

    downloaded_count = len(list(out_dir.glob("*")))  # images already on disk

    print("=" * 70)
    print(f"Character:  {character_name}")
    print(f"Query:      {query}")
    print(f"Search URL: {search_url}")
    print(f"Output:     {out_dir}")
    print(f"Already downloaded: {downloaded_count} images")
    print("=" * 70)

    after = None
    seen_post_ids: set[str] = set()
    total_posts = 0

    with open(meta_path, "a", encoding="utf-8") as meta_f:
        while downloaded_count < MAX_IMAGES:
            remaining = MAX_IMAGES - downloaded_count
            try:
                data = reddit_search(
                    session,
                    query,
                    after=after,
                    limit=min(150, remaining * 3),
                    sort=SORT,
                    t=TIME_FILTER,
                )
            except Exception as e:
                print(f"  Search error: {e}", file=sys.stderr)
                break

            children = data.get("data", {}).get("children", [])
            if not children:
                print("  No more results.")
                break

            for child in children:
                if downloaded_count >= MAX_IMAGES:
                    break

                post = child.get("data", {})
                post_id = post.get("id")
                if not post_id or post_id in seen_post_ids:
                    continue
                seen_post_ids.add(post_id)
                total_posts += 1

                image_items = extract_image_urls(post)
                title = post.get("title", "")
                print(
                    f"  [{total_posts}] r/{post.get('subreddit')} score={post.get('score')} "
                    f"— {len(image_items)} image(s) — {title[:60]}"
                )

                for img_idx, item in enumerate(image_items, start=1):
                    if downloaded_count >= MAX_IMAGES:
                        break

                    url = item["url"]
                    if url in seen_urls:
                        print(f"    dup: {url[:80]}")
                        continue

                    seen_urls.add(url)

                    ext = Path(urlparse(url).path).suffix.lower() or ".jpg"
                    filename = (
                        sanitize_filename(
                            f"{downloaded_count + 1:05d}_{post_id}_{img_idx}_{item['kind']}"
                        )
                        + ext
                    )
                    out_path = out_dir / filename

                    record = {
                        "character": character_name,
                        "slug": slug,
                        "query": query,
                        "sort": SORT,
                        "time_filter": TIME_FILTER,
                        "post_id": post_id,
                        "subreddit": post.get("subreddit"),
                        "title": title,
                        "author": post.get("author"),
                        "created_utc": post.get("created_utc"),
                        "score": post.get("score"),
                        "permalink": "https://www.reddit.com"
                        + post.get("permalink", ""),
                        "media_url": url,
                        "media_kind": item["kind"],
                        "local_path": str(out_path),
                    }

                    if out_path.exists():
                        print(f"    exists: {out_path.name}")
                        downloaded_count += 1
                    else:
                        try:
                            digest = download_file(session, url, out_path)
                            if digest in seen_hashes:
                                # Duplicate content from a cross-post — discard
                                out_path.unlink(missing_ok=True)
                                record["content_dup_hash"] = digest
                                print(
                                    f"    content-dup (hash {digest[:12]}…): {url[:80]}"
                                )
                            else:
                                seen_hashes.add(digest)
                                record["sha256"] = digest
                                downloaded_count += 1
                                print(
                                    f"    saved [{downloaded_count}/{MAX_IMAGES}]: {out_path.name}"
                                )
                            time.sleep(SLEEP_BETWEEN_DOWNLOADS)
                        except Exception as e:
                            record["download_error"] = str(e)
                            print(f"    FAILED: {url[:80]} — {e}", file=sys.stderr)

                    meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    meta_f.flush()

            # Persist seen URLs and hashes after each page
            with open(seen_path, "w", encoding="utf-8") as f:
                json.dump(sorted(seen_urls), f)
            with open(hashes_path, "w", encoding="utf-8") as f:
                json.dump(sorted(seen_hashes), f)

            after = data.get("data", {}).get("after")
            if not after:
                print("  No next page token.")
                break

            time.sleep(SLEEP_BETWEEN_PAGES)

    print(f"Done: {downloaded_count} images for {character_name}.")


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    if not QUALIFYING_CHARS_PATH.exists():
        print(f"ERROR: {QUALIFYING_CHARS_PATH} not found.")
        print("Run wikimedia_character_pipeline.py first, or create the file manually.")
        sys.exit(1)

    with open(QUALIFYING_CHARS_PATH, encoding="utf-8") as f:
        characters = json.load(f)

    print(
        f"Loaded {len(characters)} qualifying characters from {QUALIFYING_CHARS_PATH}"
    )

    sess = requests.Session()

    for i, char in enumerate(characters, start=1):
        name = char["character_name"]
        slug = char["slug"]
        print(f"\n\n##### [{i}/{len(characters)}] {name} #####\n")
        try:
            download_character_reddit(name, slug, sess)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            sys.exit(0)
        except Exception as e:
            print(f"Unexpected error for {name}: {e}", file=sys.stderr)

        if i < len(characters):
            time.sleep(SLEEP_BETWEEN_CHARACTERS)

    print("\nReddit pipeline complete.")


if __name__ == "__main__":
    main()
