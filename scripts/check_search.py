#!/usr/bin/env python3
"""Check generated search coverage and every section destination before deployment."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urljoin
import argparse
import json
from check_site import SiteChecker


class SearchMetadata(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.language = 'en'
        self.url = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'html':
            self.language = attrs.get('lang', 'en')
        if tag == 'article' and attrs.get('data-search-include') == 'true':
            self.url = attrs.get('data-search-url')


def check(root):
    checker = SiteChecker(root)
    expected = {language: set() for language in ('en', 'ja', 'ko')}
    for path in root.rglob('*.html'):
        metadata = SearchMetadata(path.read_text(encoding='utf-8'))
        if metadata.url:
            expected.setdefault(metadata.language, set()).add(metadata.url)
    failures = []
    count = 0
    for language, source_urls in expected.items():
        path = root / f'search-{language}.json'
        try:
            articles = json.loads(path.read_text(encoding='utf-8'))
            actual = {article['url'] for article in articles}
            if actual != source_urls or len(actual) != len(articles):
                failures.append(f'{path.name}: missing, extra or duplicate article entries')
            for article in articles:
                if not article.get('title') or not article.get('sections'):
                    failures.append(f'{path.name}: empty article {article.get("url")}')
                for section in article['sections']:
                    count += 1
                    url = urljoin(checker.site_url, section['url'])
                    error = checker.destination_error(url)
                    if not checker.local(url):
                        error = 'external search destination'
                    if not section['text'].strip():
                        error = 'empty searchable passage'
                    if error:
                        failures.append(f'{path.name}: {section["url"]}: {error}')
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(f'{path.name}: {error}')
    for failure in failures[:30]:
        print(failure)
    print(f'Checked {count} search sections in {len(expected)} languages; {len(failures)} failures.')
    return 1 if failures else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('site_dir', type=Path)
    raise SystemExit(check(parser.parse_args().site_dir))
