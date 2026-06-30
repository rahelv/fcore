#!/usr/bin/env python3
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, quote_plus

import requests


USER_AGENT = "CostumeRecognitionBot/0.1 by u/your_reddit_username"

OUTDIR = Path("")
MAX_POSTS = 150
SORT = "relevance"
TIME_FILTER = "all"
SLEEP_BETWEEN_DOWNLOADS = 1.0
SLEEP_BETWEEN_PAGES = 1.0
SLEEP_BETWEEN_QUERIES = 2.0

CHARACTERS = [
    # "Jinx",
    "Vi",
    "Gojo Satoru",
    "Yuji Itadori",
    "Raiden Shogun",
    "Hu Tao",
    "Furina",
    "Kafka",
    "Firefly Honkai Star Rail",
    "March 7th",
]


def sanitize_filename(name: str, max_len: int = 180) -> str:
    name = re.sub(r"[^\w\-. ]+", "_", name, flags=re.UNICODE).strip()
    name = re.sub(r"\s+", " ", name)
    if not name:
        name = "file"
    return name[:max_len].rstrip(" .")


def build_search_url(query: str) -> str:
    return f"https://www.reddit.com/search/?q={quote_plus(query)}&type=media"


def reddit_search(query: str, after: str | None = None, limit: int = 100,
                  sort: str = "relevance", t: str = "all"):
    url = "https://www.reddit.com/search/.json"
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

    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def pick_extension_from_url(url: str, fallback: str = ".bin") -> str:
    path = urlparse(url).path.lower()
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4", ".mov", ".gifv"]:
        if path.endswith(ext):
            return ext
    return fallback


def download_file(session: requests.Session, url: str, out_path: Path, sleep_s: float = 1.0):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": USER_AGENT}
    with session.get(url, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

    time.sleep(sleep_s)


def extract_media_from_post(post: dict):
    items = []

    if post.get("is_gallery") and post.get("media_metadata"):
        media_metadata = post["media_metadata"]
        gallery_data = post.get("gallery_data", {})
        for i, item in enumerate(gallery_data.get("items", []), start=1):
            media_id = item.get("media_id")
            meta = media_metadata.get(media_id, {})
            if not meta:
                continue

            s = meta.get("s", {})
            url = s.get("u") or s.get("gif") or s.get("mp4")
            if url:
                url = url.replace("&amp;", "&")
                items.append({
                    "url": url,
                    "kind": "gallery",
                    "label": f"gallery_{i}",
                })

    preview = post.get("preview", {})
    images = preview.get("images", [])
    if images:
        src = images[0].get("source", {})
        url = src.get("url")
        if url:
            items.append({
                "url": url.replace("&amp;", "&"),
                "kind": "image",
                "label": "preview",
            })

    media = post.get("media") or {}
    reddit_video = media.get("reddit_video") or {}
    fallback_url = reddit_video.get("fallback_url")
    if fallback_url:
        items.append({
            "url": fallback_url,
            "kind": "video",
            "label": "reddit_video",
        })

    secure_media = post.get("secure_media") or {}
    reddit_video2 = secure_media.get("reddit_video") or {}
    fallback_url2 = reddit_video2.get("fallback_url")
    if fallback_url2:
        items.append({
            "url": fallback_url2,
            "kind": "video",
            "label": "secure_reddit_video",
        })

    url = post.get("url_overridden_by_dest") or post.get("url")
    if url:
        lower = url.lower()
        if any(ext in lower for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4", ".mov"]):
            items.append({
                "url": url,
                "kind": "direct",
                "label": "direct_url",
            })

    seen = set()
    deduped = []
    for item in items:
        if item["url"] not in seen:
            deduped.append(item)
            seen.add(item["url"])

    return deduped


def download_character_search(character: str, session: requests.Session):
    query = f"{character} cosplay"
    search_url = build_search_url(query)

    media_dir = OUTDIR / "global_search" / sanitize_filename(query)
    metadata_path = media_dir / "metadata.jsonl"
    media_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Character: {character}")
    print(f"Query: {query}")
    print(f"Search URL: {search_url}")
    print(f"Saving to: {media_dir}")
    print("=" * 80)

    after = None
    seen_posts = set()
    total_posts = 0
    total_downloads = 0

    with open(metadata_path, "a", encoding="utf-8") as meta_f:
        while True:
            remaining = MAX_POSTS - total_posts
            if remaining <= 0:
                break

            limit = min(100, remaining)

            try:
                data = reddit_search(
                    query=query,
                    after=after,
                    limit=limit,
                    sort=SORT,
                    t=TIME_FILTER,
                )
            except Exception as e:
                print(f"Search failed for '{character}': {e}", file=sys.stderr)
                break

            listing = data.get("data", {})
            children = listing.get("children", [])

            if not children:
                print("No more search results.")
                break

            for rank_on_page, child in enumerate(children, start=1):
                post = child.get("data", {})
                post_id = post.get("id")
                if not post_id or post_id in seen_posts:
                    continue
                seen_posts.add(post_id)
                total_posts += 1

                title = post.get("title", "")
                permalink = "https://www.reddit.com" + post.get("permalink", "")
                created_utc = post.get("created_utc")
                author = post.get("author")
                post_subreddit = post.get("subreddit")
                score = post.get("score")
                media_items = extract_media_from_post(post)

                print(f"[{total_posts}] r/{post_subreddit} - score={score} - {title} -> {len(media_items)} media item(s)")

                for idx, item in enumerate(media_items, start=1):
                    media_url = item["url"]
                    ext = pick_extension_from_url(media_url)
                    filename = sanitize_filename(f"{total_posts:05d}_{post_id}_{idx}_{item['label']}") + ext
                    out_path = media_dir / filename

                    record = {
                        "character": character,
                        "search_url": search_url,
                        "search_rank": total_posts,
                        "rank_on_page": rank_on_page,
                        "post_id": post_id,
                        "mode": "global",
                        "subreddit": post_subreddit,
                        "query": query,
                        "sort": SORT,
                        "time_filter": TIME_FILTER,
                        "title": title,
                        "author": author,
                        "created_utc": created_utc,
                        "score": score,
                        "permalink": permalink,
                        "post_url": post.get("url"),
                        "media_url": media_url,
                        "media_kind": item["kind"],
                        "local_path": str(out_path),
                    }

                    if out_path.exists():
                        print(f"    exists: {out_path.name}")
                    else:
                        try:
                            print(f"    downloading: {media_url}")
                            download_file(session, media_url, out_path, sleep_s=SLEEP_BETWEEN_DOWNLOADS)
                            total_downloads += 1
                        except Exception as e:
                            print(f"    failed: {media_url} ({e})", file=sys.stderr)
                            record["download_error"] = str(e)

                    meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    meta_f.flush()

                if total_posts >= MAX_POSTS:
                    print(f"Reached max posts limit: {MAX_POSTS}")
                    break

            after = listing.get("after")
            if not after or total_posts >= MAX_POSTS:
                print("No further page token." if not after else "Finished due to max posts.")
                break

            time.sleep(SLEEP_BETWEEN_PAGES)

    print(f"Done for {character}. Inspected {total_posts} post(s), downloaded {total_downloads} file(s).")
    print(f"Metadata written to: {metadata_path}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    for i, character in enumerate(CHARACTERS, start=1):
        try:
            print(f"\n\n##### [{i}/{len(CHARACTERS)}] STARTING: {character} #####\n")
            download_character_search(character, session)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error for '{character}': {e}", file=sys.stderr)

        if i < len(CHARACTERS):
            time.sleep(SLEEP_BETWEEN_QUERIES)


if __name__ == "__main__":
    main()