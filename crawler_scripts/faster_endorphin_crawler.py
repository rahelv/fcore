from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse
import requests

from playwright.sync_api import sync_playwright

BASE_URL = "https://de.endorphin.ch/2022/polymanga"
START_NUM = 1
END_NUM = 502
OUT_DIR = Path("endorphin_polymanga_2022_images")
HEADLESS = True
DOWNLOAD_WORKERS = 8


def filename_from_url(url: str, fallback: str) -> str:
    name = Path(urlparse(url).path).name
    return name or fallback


def download_file(
    session: requests.Session, url: str, out_dir: Path, fallback_name: str
) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = filename_from_url(url, fallback_name)
    out_path = out_dir / filename

    if out_path.exists():
        return f"[skip] {out_path.name}"

    with session.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    return f"[saved] {out_path.name}"


def extract_best_image(page):
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

            const imgs = [...document.querySelectorAll('img')]
                .filter(isVisible)
                .map(img => {
                    const rect = img.getBoundingClientRect();
                    return {
                        url: abs(img.currentSrc || img.src),
                        area: rect.width * rect.height
                    };
                })
                .filter(x => x.url)
                .sort((a, b) => b.area - a.area);

            if (imgs.length) return imgs[0].url;

            return null;
        }
        """
    )


def collect_image_urls():
    found = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        page = browser.new_page(viewport={"width": 1600, "height": 1200})

        print(f"[open] {BASE_URL}", flush=True)
        page.goto(BASE_URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(1000)

        for num in range(START_NUM, END_NUM + 1):
            print(f"[collect {num}/{END_NUM}]", flush=True)

            try:
                page.evaluate(
                    "(hash) => { window.location.hash = hash; }", f"nummer-{num}"
                )
                page.wait_for_timeout(250)

                image_url = extract_best_image(page)

                if image_url:
                    found.append((num, image_url))
                    print(f"  [found] {image_url}", flush=True)
                else:
                    print("  [warn] no image found", flush=True)

            except Exception as e:
                print(f"  [failed] nummer-{num} -> {e}", flush=True)

        browser.close()

    return found


def download_all(found):
    unique = {}
    for num, url in found:
        unique.setdefault(url, num)

    print(f"[download] {len(unique)} unique images", flush=True)

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=DOWNLOAD_WORKERS, pool_maxsize=DOWNLOAD_WORKERS
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = {
            executor.submit(
                download_file, session, url, OUT_DIR, f"nummer-{num}.jpg"
            ): (num, url)
            for url, num in unique.items()
        }

        for future in as_completed(futures):
            num, url = futures[future]
            try:
                result = future.result()
                print(f"[{num}] {result}", flush=True)
            except Exception as e:
                print(f"[{num}] [failed] {url} -> {e}", flush=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print(f"Base URL: {BASE_URL}", flush=True)
    print(f"Range: {START_NUM} -> {END_NUM}", flush=True)
    print(f"Headless: {HEADLESS}", flush=True)
    print(f"Workers: {DOWNLOAD_WORKERS}", flush=True)
    print("=" * 70, flush=True)

    found = collect_image_urls()
    print(f"[collected] {len(found)} image references", flush=True)

    download_all(found)

    print(f"[done] saved into: {OUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
