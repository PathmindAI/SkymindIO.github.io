(() => {
  const browser = document.querySelector('.article-browser');
  if (browser && window.matchMedia('(max-width: 900px)').matches) browser.open = false;

  const input = document.querySelector('#wiki-search-input');
  if (input) {
    const rows = [...document.querySelectorAll('#wiki-links-ul li')];
    const empty = document.querySelector('.search-empty');
    const status = document.querySelector('.search-status');
    input.addEventListener('input', () => {
      const query = input.value.trim().toLocaleLowerCase();
      let matches = 0;
      for (const row of rows) {
        row.hidden = !row.textContent.toLocaleLowerCase().includes(query);
        if (!row.hidden) matches += 1;
      }
      empty.hidden = matches !== 0;
      status.hidden = !query;
      status.textContent = `${matches} ${matches === 1 ? 'topic' : 'topics'} found`;
    });
  }

  // Keep embedded media working without the old theme runtime.
  for (const element of document.querySelectorAll('[data-lazy-src]')) {
    element.loading = 'lazy';
    element.src = element.dataset.lazySrc;
  }
})();
