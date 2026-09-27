((root) => {
  'use strict';

  function termsFor(query) {
    return [...new Set((query.toLocaleLowerCase().match(/[\p{L}\p{N}]+/gu) || []))];
  }

  function excerpt(text, terms, length = 180) {
    if (text.length <= length) return text;
    const lower = text.toLocaleLowerCase();
    const matches = terms.map(term => lower.indexOf(term)).filter(index => index >= 0);
    const first = matches.length ? Math.min(...matches) : 0;
    let start = Math.max(0, first - 55);
    if (start > 0) {
      const boundary = text.indexOf(' ', start);
      if (boundary >= 0 && boundary < first) start = boundary + 1;
    }
    let end = Math.min(text.length, start + length);
    if (end < text.length) {
      const boundary = text.lastIndexOf(' ', end);
      if (boundary > start + length / 2) end = boundary;
    }
    return (start ? '…' : '') + text.slice(start, end).trim() + (end < text.length ? '…' : '');
  }

  function search(articles, query) {
    const terms = termsFor(query);
    if (!terms.length) return [];
    const phrase = query.trim().toLocaleLowerCase();
    const results = [];
    for (const article of articles) {
      const title = `${article.title} ${article.fullTitle || ''}`.toLocaleLowerCase();
      let best = null;
      for (const section of article.sections) {
        const heading = (section.title || '').toLocaleLowerCase();
        const body = section.text.toLocaleLowerCase();
        const all = `${title} ${heading} ${body}`;
        if (!terms.every(term => all.includes(term))) continue;
        let score = terms.reduce((sum, term) => sum + (title.includes(term) ? 12 : 0) +
          (heading.includes(term) ? 4 : 0) + (body.includes(term) ? 1 : 0), 0);
        if (title.includes(phrase)) score += 8;
        if (body.includes(phrase)) score += 3;
        if (!best || score > best.score) {
          best = {title: article.title, url: section.url || article.url,
            heading: section.title || '', snippet: excerpt(section.text, terms), score};
        }
      }
      if (best) results.push(best);
    }
    return results.sort((a, b) => b.score - a.score || a.title.localeCompare(b.title));
  }

  const api = {termsFor, excerpt, search};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.WikiSearch = api;
})(typeof window !== 'undefined' ? window : globalThis);
