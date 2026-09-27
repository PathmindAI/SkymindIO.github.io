# AI Wiki

The site is served at https://wiki.pathmind.com/. English guides live in `_wiki`; Japanese and Korean guides live in `_wiki-jp` and `_wiki-kr`.

## Publication checks

The **Check and publish wiki** workflow builds the site, generates search indexes and checks links before deployment. A failed check prevents that build from publishing. Pull requests run the same checks without deploying. GitHub Pages must use **GitHub Actions** as its publishing source.

The link checker covers English articles, the homepage, the personal page and shared navigation. It follows the HTML redirects to verify their destinations, including section anchors. Translated article bodies and external websites are outside that check. Search destinations are checked in all three languages.

After building with Jekyll, run:

```sh
python3 -m unittest discover -s scripts -p 'test_*.py'
node --test scripts/search.test.cjs
python3 scripts/build_search.py _site
python3 scripts/check_site.py _site
python3 scripts/check_search.py _site
```

## Search and old addresses

Search indexes are generated from rendered article text, excluding navigation and author notes. Results link to the best matching section. The browser downloads the index for the current language when a reader focuses the search field.

A retired address with a close replacement has a standalone redirect page, excluded from the sitemap. Keep merged articles pointing to their replacement section. Other missing addresses use `404.html`, which sends visitors to the index while retaining the initial HTTP 404 response. These are HTML redirects; GitHub Pages does not provide per-path server redirect rules in this repository.
