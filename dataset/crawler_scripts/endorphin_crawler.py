import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from playwright.sync_api import sync_playwright

BASE_URL = "https://de.endorphin.ch/2023/herofest"
START_NUM = 1
END_NUM = 265
OUT_DIR = Path("herofest_2023_images")
HEADLESS = False


def filename_from_url(url: str, fallback: str) -> str:
    name = Path(urlparse(url).path).name
    return name or fallback


def download_file(url: str, out_dir: Path, fallback_name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = filename_from_url(url, fallback_name)
    out_path = out_dir / filename

    if out_path.exists():
        print(f"  [skip] already exists: {out_path.name}", flush=True)
        return out_path

    print(f"  [download] {url}", flush=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"  [saved] {out_path}", flush=True)
    return out_path


def extract_best_image(page):
    """
    Try several strategies and return the best candidate image URL.
    Priority:
    1) visible anchors directly linking to image files
    2) visible large img elements
    3) any image-looking URL found in visible elements
    """
    return page.evaluate(
        """
        () => {
            function isVisible(el) {
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return (
                    style.display !== 'none' &&
                    style.visibility !== 'hidden' &&
                    style.opacity !== '0' &&
                    rect.width > 40 &&
                    rect.height > 40
                );
            }

            function abs(u) {
                try { return new URL(u, location.href).href; }
                catch { return null; }
            }

            function looksLikeImage(u) {
                if (!u) return false;
                return /\\.(jpg|jpeg|png|webp|gif|avif)(\\?|#|$)/i.test(u);
            }

            // Strategy 1: visible links directly to image files
            const visibleImageLinks = [...document.querySelectorAll('a[href]')]
                .filter(isVisible)
                .map(a => abs(a.getAttribute('href')))
                .filter(looksLikeImage);

            if (visibleImageLinks.length) {
                return {
                    url: visibleImageLinks[0],
                    source: "visible-image-link"
                };
            }

            // Strategy 2: visible images, choose the largest one
            const imgs = [...document.querySelectorAll('img')]
                .filter(isVisible)
                .map(img => {
                    const rect = img.getBoundingClientRect();
                    return {
                        url: abs(img.currentSrc || img.src),
                        area: rect.width * rect.height,
                        width: rect.width,
                        height: rect.height,
                        alt: img.alt || ""
                    };
                })
                .filter(x => x.url);

            imgs.sort((a, b) => b.area - a.area);

            if (imgs.length) {
                return {
                    url: imgs[0].url,
                    source: "largest-visible-img",
                    width: imgs[0].width,
                    height: imgs[0].height,
                    alt: imgs[0].alt
                };
            }

            // Strategy 3: check background images on visible elements
            const bgCandidates = [];
            for (const el of document.querySelectorAll('*')) {
                if (!isVisible(el)) continue;
                const bg = window.getComputedStyle(el).backgroundImage;
                if (!bg || bg === 'none') continue;

                const matches = [...bg.matchAll(/url\\(["']?(.*?)["']?\\)/g)];
                for (const m of matches) {
                    const u = abs(m[1]);
                    if (u) bgCandidates.push(u);
                }
            }

            if (bgCandidates.length) {
                return {
                    url: bgCandidates[0],
                    source: "background-image"
                };
            }

            return null;
        }
        """
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    failed = 0
    seen_urls = set()

    print("=" * 70, flush=True)
    print(f"Base URL:   {BASE_URL}", flush=True)
    print(f"Range:      {START_NUM} -> {END_NUM}", flush=True)
    print(f"Output dir: {OUT_DIR.resolve()}", flush=True)
    print("=" * 70, flush=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        page = browser.new_page(viewport={"width": 1600, "height": 1200})

        print(f"[open] {BASE_URL}", flush=True)
        page.goto(BASE_URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(2000)

        for num in range(START_NUM, END_NUM + 1):
            target = f"{BASE_URL}#nummer-{num}"
            print(f"\n[{num}/{END_NUM}] {target}", flush=True)

            try:
                # Set the hash directly
                page.evaluate(
                    "(hash) => { window.location.hash = hash; }", f"nummer-{num}"
                )
                page.wait_for_timeout(1200)

                # Small scroll jiggle in case lazy-loading depends on viewport updates
                page.evaluate(
                    """
                    () => {
                        window.scrollBy(0, 120);
                        window.scrollBy(0, -120);
                    }
                    """
                )
                page.wait_for_timeout(500)

                candidate = extract_best_image(page)

                if not candidate or not candidate.get("url"):
                    print("  [warn] no image found for this number", flush=True)
                    failed += 1
                    continue

                image_url = candidate["url"]
                print(f"  [found] {image_url}", flush=True)
                print(f"  [via]   {candidate.get('source', 'unknown')}", flush=True)

                if image_url in seen_urls:
                    print("  [skip] duplicate image URL", flush=True)
                    continue

                fallback_name = f"nummer-{num}.jpg"
                download_file(image_url, OUT_DIR, fallback_name)
                seen_urls.add(image_url)
                downloaded += 1

            except Exception as e:
                print(f"  [failed] nummer-{num} -> {e}", flush=True)
                failed += 1

        browser.close()

    print("\n" + "=" * 70, flush=True)
    print(f"Done.", flush=True)
    print(f"Downloaded: {downloaded}", flush=True)
    print(f"Failed:     {failed}", flush=True)
    print(f"Unique URLs:{len(seen_urls)}", flush=True)
    print(f"Saved to:   {OUT_DIR.resolve()}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
