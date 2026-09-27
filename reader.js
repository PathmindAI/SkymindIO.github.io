(() => {
  const browser = document.querySelector('.article-browser');
  if (browser && window.matchMedia('(max-width: 900px)').matches) browser.open = false;

  const input = document.querySelector('#wiki-search-input');
  if (input) {
    const rows = [...document.querySelectorAll('#wiki-links-ul li')];
    const topics = document.querySelector('.topic-navigation');
    const resultsNav = document.querySelector('.search-navigation');
    const resultsList = document.querySelector('#wiki-search-results');
    const empty = document.querySelector('.search-empty');
    const status = document.querySelector('.search-status');
    let articles;
    let loading;
    let timer;
    let pendingRender = false;

    function linkHasFocus() {
      return topics.contains(document.activeElement) || resultsNav.contains(document.activeElement);
    }

    function deferForFocus() {
      pendingRender = linkHasFocus();
      return pendingRender;
    }

    function showTitles(message) {
      if (deferForFocus()) return;
      const query = input.value.trim().toLocaleLowerCase();
      let count = 0;
      for (const row of rows) {
        row.hidden = !row.textContent.toLocaleLowerCase().includes(query);
        if (!row.hidden) count += 1;
      }
      topics.hidden = false;
      resultsNav.hidden = true;
      empty.hidden = !query || count > 0 || Boolean(message);
      status.hidden = !query;
      status.textContent = query ? (message || `${count} title matches`) : '';
    }

    function appendHighlighted(parent, text, terms) {
      const lower = text.toLocaleLowerCase();
      let offset = 0;
      while (offset < text.length) {
        let start = text.length;
        let match = '';
        for (const term of terms) {
          const position = lower.indexOf(term, offset);
          if (position >= 0 && (position < start || (position === start && term.length > match.length))) {
            start = position;
            match = term;
          }
        }
        parent.append(document.createTextNode(text.slice(offset, start)));
        if (!match) break;
        const mark = document.createElement('mark');
        mark.textContent = text.slice(start, start + match.length);
        parent.append(mark);
        offset = start + match.length;
      }
    }

    function showResults() {
      if (deferForFocus()) return;
      const query = input.value.trim();
      if (!query) { showTitles(); return; }
      const results = window.WikiSearch.search(articles, query);
      const terms = window.WikiSearch.termsFor(query);
      resultsList.replaceChildren();
      for (const result of results) {
        const row = document.createElement('li');
        const link = document.createElement('a');
        link.href = result.url;
        const title = document.createElement('span');
        title.className = 'search-result-title';
        appendHighlighted(title, result.title, terms);
        link.append(title);
        if (result.heading && result.heading !== result.title) {
          const heading = document.createElement('span');
          heading.className = 'search-result-heading';
          heading.textContent = result.heading;
          link.append(heading);
        }
        const snippet = document.createElement('span');
        snippet.className = 'search-result-excerpt';
        appendHighlighted(snippet, result.snippet, terms);
        link.append(snippet);
        row.append(link);
        resultsList.append(row);
      }
      topics.hidden = true;
      resultsNav.hidden = results.length === 0;
      empty.hidden = results.length !== 0;
      status.hidden = false;
      status.textContent = `${results.length} ${results.length === 1 ? 'article' : 'articles'} found`;
    }

    async function loadIndex() {
      if (articles) return;
      if (!loading) {
        loading = fetch(input.dataset.searchIndex).then(response => {
          if (!response.ok) throw new Error('Search index unavailable');
          return response.json();
        }).then(data => {
          if (!Array.isArray(data) || !data.every(article => typeof article.title === 'string' &&
              typeof article.url === 'string' && Array.isArray(article.sections))) {
            throw new Error('Invalid search index');
          }
          articles = data;
        }).catch(error => { loading = null; throw error; });
      }
      return loading;
    }

    async function update() {
      if (!input.value.trim()) { showTitles(); return; }
      if (!articles) showTitles('Searching article text…');
      try {
        await loadIndex();
        showResults();
      } catch (_) {
        showTitles('Full-text search is unavailable. Showing title matches.');
      }
    }

    input.addEventListener('focus', () => { loadIndex().catch(() => {}); });
    browser.addEventListener('focusout', () => {
      if (pendingRender) setTimeout(() => { if (!linkHasFocus()) update(); }, 0);
    });
    input.addEventListener('input', () => {
      clearTimeout(timer);
      if (!input.value.trim()) showTitles();
      else timer = setTimeout(update, 120);
    });
    input.addEventListener('keydown', event => {
      if (event.key === 'Escape') {
        input.value = '';
        clearTimeout(timer);
        showTitles();
      }
    });
  }

  for (const element of document.querySelectorAll('[data-lazy-src]')) {
    element.loading = 'lazy';
    element.src = element.dataset.lazySrc;
  }
})();
