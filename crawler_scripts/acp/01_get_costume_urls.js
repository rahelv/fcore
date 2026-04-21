(async () => {
  function setPage(url, pageNum) {
    const u = new URL(url);
    u.searchParams.set("page", String(pageNum));
    return u.toString();
  }

  function getTotalPages(doc) {
    const nums = [];
    for (const a of doc.querySelectorAll("div.pagination a[href]")) {
      const txt = a.textContent.trim();
      if (/^\d+$/.test(txt)) nums.push(Number(txt));

      try {
        const u = new URL(a.href, location.origin);
        const p = u.searchParams.get("page");
        if (p && /^\d+$/.test(p)) nums.push(Number(p));
      } catch {}
    }
    return nums.length ? Math.max(...nums) : 1;
  }

  function extractFromDoc(doc, baseUrl) {
    const out = [];
    const seen = new Set();
    const ul = doc.querySelector("ul.thumbnails.largethumbs.medium");
    if (!ul) return out;

    for (const li of ul.children) {
      let a = li.querySelector('div.thumbdescription a[href*="display.php?c="]');
      if (!a) a = li.querySelector('a[href*="display.php?c="]');
      if (!a) continue;

      const text = a.textContent.trim().replace(/\s+/g, " ");
      const url = new URL(a.getAttribute("href"), baseUrl).href;

      if (!seen.has(url)) {
        seen.add(url);
        out.push({ url });
      }
    }
    return out;
  }

  const startUrl = location.href;
  const totalPages = getTotalPages(document);
  console.log("Detected pages:", totalPages);

  const all = [];
  const globalSeen = new Set();

  for (let page = 1; page <= totalPages; page++) {
    const pageUrl = setPage(startUrl, page);
    console.log("Fetching", pageUrl);

    const res = await fetch(pageUrl, { credentials: "include" });
    const html = await res.text();

    const doc = new DOMParser().parseFromString(html, "text/html");
    const items = extractFromDoc(doc, pageUrl);

    console.log(`Page ${page}: ${items.length} links`);

    for (const item of items) {
      if (!globalSeen.has(item.url)) {
        globalSeen.add(item.url);
        all.push(item);
      }
    }
  }

  const output = all.map(x => `"${x.url}",`).join("\n");

  console.table(all);
  console.log("----- COPY FROM BELOW -----");
  console.log(output);
  console.log("----- END -----");

  try {
    await navigator.clipboard.writeText(output);
    console.log(`Copied ${all.length} total unique links to clipboard`);
  } catch (e) {
    console.log("Clipboard write failed, but output is printed above.");
  }

  all;
})();