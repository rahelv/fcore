import os
from pathlib import Path
from urllib.parse import urlparse

import requests
from playwright.sync_api import sync_playwright

PAGE_URL = "https://fantasybasel.ch/en/2025"
OUT_DIR = Path("fantasybasel_2025_images")


def filename_from_url(url: str) -> str:
    return Path(urlparse(url).path).name or "image.jpg"


def download_file(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = filename_from_url(url)
    out_path = out_dir / filename

    if out_path.exists():
        print(f"[skip] {filename}", flush=True)
        return out_path

    print(f"[downloading] {url}", flush=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"[saved] {out_path}", flush=True)
    return out_path


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page(viewport={"width": 1400, "height": 1600})

        print(f"[open] {PAGE_URL}", flush=True)
        page.goto(PAGE_URL, wait_until="networkidle", timeout=60000)

        # Scroll a bit in case anything lazy-loads
        page.evaluate("""
            async () => {
                for (let i = 0; i < 6; i++) {
                    window.scrollTo(0, document.body.scrollHeight);
                    await new Promise(r => setTimeout(r, 500));
                }
                window.scrollTo(0, 0);
            }
        """)

        print("[collect] finding gallery image links...", flush=True)

        urls = page.eval_on_selector_all(
            'a[href*="/public/images/gallery/"]',
            """els => els
                .map(a => a.href)
                .filter(Boolean)
            """,
        )

        # fallback: also inspect img/src in case some are direct image nodes
        img_urls = page.eval_on_selector_all(
            'img[src*="/public/images/gallery/"]',
            """els => els
                .map(img => img.currentSrc || img.src)
                .filter(Boolean)
            """,
        )

        browser.close()

    all_urls = sorted(set(urls + img_urls))

    print(f"[found] {len(all_urls)} unique image URLs", flush=True)
    for i, url in enumerate(all_urls, 1):
        print(f"  {i:03d}. {url}", flush=True)

    for i, url in enumerate(all_urls, 1):
        print(f"[{i}/{len(all_urls)}]", flush=True)
        try:
            download_file(url, OUT_DIR)
        except Exception as e:
            print(f"[failed] {url} -> {e}", flush=True)

    print(f"[done] saved into: {OUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
