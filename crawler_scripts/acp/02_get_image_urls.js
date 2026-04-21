(async () => {
  const pages = [
    // INSERT LINKS HERE
  ];

  const STORE_KEY = 'acp_scrape_state_v1';

  const saved = JSON.parse(localStorage.getItem(STORE_KEY) || '{}');

  const state = {
    stop: false,
    results: saved.results || [],
    donePages: saved.donePages || [],
    failures: saved.failures || [],
    currentPage: null
  };

  const saveState = () => {
    localStorage.setItem(STORE_KEY, JSON.stringify({
      results: state.results,
      donePages: state.donePages,
      failures: state.failures,
      currentPage: state.currentPage,
      savedAt: new Date().toISOString()
    }));
  };

  window.scrapeState = state;

  const MIN_DELAY_MS = 5000;
  const MAX_DELAY_MS = 9000;
  const RETRY_DELAY_MS = 20000;
  const MAX_RETRIES = 3;

  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const randomDelay = () =>
    Math.floor(Math.random() * (MAX_DELAY_MS - MIN_DELAY_MS + 1)) + MIN_DELAY_MS;

  const toFullImageUrl = src =>
    src.replace(/-t(?=\.[a-zA-Z0-9]+(?:\?|$))/, '');

  const remainingPages = pages.filter(url => !state.donePages.includes(url));

  console.log(`Resuming with ${remainingPages.length} pages left`);

  for (let i = 0; i < remainingPages.length; i++) {
    if (state.stop) {
      console.warn('Stopped by user');
      break;
    }

    const pageUrl = remainingPages[i];
    state.currentPage = pageUrl;
    saveState();

    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      if (state.stop) break;

      try {
        const waitMs = attempt === 1 ? randomDelay() : RETRY_DELAY_MS;
        console.log(`Waiting ${waitMs}ms before ${pageUrl}`);
        await sleep(waitMs);

        const res = await fetch(pageUrl, {
          credentials: 'include',
          cache: 'no-store'
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const html = await res.text();
        const doc = new DOMParser().parseFromString(html, 'text/html');

        // Only get the main thumbnail image inside each photo entry:
        // li > a > img
        const imgs = [...doc.querySelectorAll('#photolisttabs li > a > img')]
          .map(img => img.getAttribute('src'))
          .filter(Boolean)
          .map(src => new URL(src, pageUrl).href)
          .map(toFullImageUrl);

        state.results.push(...imgs);
        state.donePages.push(pageUrl);
        state.currentPage = null;
        saveState();

        console.log(`Saved ${imgs.length} images from ${pageUrl}`);
        break;
      } catch (err) {
        console.warn(`Failed ${pageUrl} attempt ${attempt}: ${err.message}`);
        if (attempt === MAX_RETRIES) {
          state.failures.push(pageUrl);
          state.currentPage = null;
          saveState();
        }
      }
    }
  }

  const unique = [...new Set(state.results)];

  // final formatted output: "url",
  const formatted = unique.map(url => `"${url}",`).join('\n');

  console.log(formatted);
  console.log(`Total unique URLs: ${unique.length}`);

  // optional: easy copy from console
  window.formattedImageList = formatted;
})();