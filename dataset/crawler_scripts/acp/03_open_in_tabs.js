const urls = [
  
];

let i = 0;
const delayMs = 1000; // increase if popup blocking happens

const timer = setInterval(() => {
  if (i >= urls.length) {
    clearInterval(timer);
    console.log(`Done. Opened ${urls.length} tabs.`);
    return;
  }

  window.open(urls[i], "_blank");
  console.log(`Opened ${i + 1}/${urls.length}: ${urls[i]}`);
  i++;
}, delayMs);