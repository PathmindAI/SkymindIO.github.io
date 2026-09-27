"""Run with: python3 -m unittest discover -s scripts -p 'test_build_search.py'."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from build_search import ArticleParser, build_indexes, extract_document


def article(body, title="Short title", url="/guide", include="true", full_title="Full guide title"):
    return (f'<article class="wiki-article" data-search-include="{include}" '
            f'data-search-title="{title}" data-search-url="{url}">'
            f'<header class="article-header"><h1>{full_title}</h1></header>'
            f'<div class="article-body">{body}</div></article>')


def documents(html):
    parser = ArticleParser()
    parser.feed(html)
    parser.close()
    return [extract_document(node) for _, node in parser.articles]


class ExtractionTests(unittest.TestCase):
    def test_excludes_layout_and_nested_noncontent(self):
        body = ('<p>First<br>line<img src="x" alt="UNWANTED image"><hr>Next.</p>'
                '<nav><h2 id="wrong">UNWANTED navigation</h2></nav>'
                '<section class="author-note"><p>UNWANTED author</p></section>'
                '<footer>UNWANTED footer</footer><script>UNWANTED script</script>'
                '<style>UNWANTED style</style><template>UNWANTED template</template>'
                '<span hidden>UNWANTED hidden</span><p>Last paragraph.</p>')
        html = '<nav>UNWANTED outside</nav>' + article(body) + '<footer>UNWANTED outside</footer>'
        document = documents(html)[0]
        self.assertEqual(document["sections"], [{
            "title": "", "text": "First line Next. Last paragraph.", "url": "/guide"}])
        self.assertNotIn("UNWANTED", json.dumps(document))

    def test_inline_markup_entities_and_unicode(self):
        body = ('<p>inter<strong>national</strong> neural <em>networks</em> '
                'A&nbsp;&amp;&nbsp;B &#169;.</p>'
                '<p>深層<strong>学習</strong>と 한국어.</p>')
        document = documents(article(body, title="A &amp; B &quot;guide&quot;"))[0]
        self.assertEqual(document["title"], 'A & B "guide"')
        self.assertEqual(document["sections"][0]["text"],
                         "international neural networks A & B ©. 深層学習と 한국어.")

    def test_heading_anchors_and_inheritance(self):
        body = ('<p>Opening.</p><h2>Unanchored first</h2><p>First.</p>'
                '<h2 id="generated"><a name="legacy">Own ID wins</a></h2><p>Second.</p>'
                '<h3>Unanchored next</h3><p>Third.</p>'
                '<h2><a name="old-anchor">Legacy <em>heading</em></a></h2><p>Fourth.</p>'
                '<h2><a id="new-anchor">New anchor</a></h2><p>Fifth.</p>'
                '<h3 id="plain-heading">Plain heading</h3><p>Sixth.</p>')
        sections = documents(article(body))[0]["sections"]
        self.assertEqual([section["url"] for section in sections], [
            "/guide", "/guide", "/guide#generated", "/guide#generated",
            "/guide#old-anchor", "/guide#new-anchor", "/guide#plain-heading"])
        self.assertEqual(sections[4]["title"], "Legacy heading")
        self.assertEqual(sections[4]["text"], "Fourth.")
        self.assertEqual(len(sections), 7)

    def test_lists_tables_and_code_keep_word_boundaries(self):
        body = ('<h2 id="example">Example</h2><ul><li>One</li><li>Two</li></ul>'
                '<table><tr><td>Cell one</td><td>Cell two</td></tr></table>'
                '<pre><code>x = 1\ny = x &lt; 2</code></pre><p>After code.</p>')
        section = documents(article(body))[0]["sections"][0]
        self.assertEqual(section["text"], "One Two Cell one Cell two x = 1 y = x < 2 After code.")

    def test_grouping_headings_keep_only_text_bearing_sections(self):
        body = ('<h2 id="group">Grouping heading</h2>'
                '<h3 id="child">Child heading</h3><p>Child passage.</p>'
                '<h2 id="next-group">Another group</h2>'
                '<h3>Unanchored child</h3><p>Inherited anchor passage.</p>'
                '<h2 id="last">Empty final heading</h2>')
        sections = documents(article(body))[0]["sections"]
        self.assertEqual(sections, [
            {"title": "Child heading", "text": "Child passage.", "url": "/guide#child"},
            {"title": "Unanchored child", "text": "Inherited anchor passage.",
             "url": "/guide#next-group"},
        ])

    def test_header_and_article_fields_stay_separate(self):
        html = (article("<h2 id='one'>Body one</h2><p>First text.</p>", "One", "/one", full_title="Long One")
                + article("<p>Second text.</p>", "Two", "/two", full_title="Long Two"))
        first, second = documents(html)
        self.assertEqual((first["title"], first["fullTitle"], first["url"]), ("One", "Long One", "/one"))
        self.assertEqual((second["title"], second["fullTitle"], second["url"]), ("Two", "Long Two", "/two"))
        self.assertEqual(second["sections"], [{"title": "", "text": "Second text.", "url": "/two"}])
        self.assertNotIn("Long One", json.dumps(first["sections"]))

    def test_empty_and_unmarked_articles_are_excluded(self):
        html = (article("<p>Home text.</p>", include="false", url="/")
                + '<article class="wiki-article"><div class="article-body">Unmarked</div></article>'
                + article("  <img src='x'><script>ignored</script> ")
                + article("<h2 id='heading-only'>Heading only</h2>", url="/heading"))
        extracted = [document for document in documents(html) if document is not None]
        self.assertEqual(extracted, [])


class IndexTests(unittest.TestCase):
    def test_languages_sorted_paths_compact_utf8_and_repeatability(self):
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory)
            samples = [
                ("z.html", "en", "Z", "/z", "Last"),
                ("a.html", "en", "A", "/a", "First"),
                ("jp/guide.html", "ja-JP", "日本語", "/jp/guide", "深層学習"),
                ("kr/guide.html", "ko", "한국어", "/kr/guide", "신경망"),
                ("other.html", "fr", "Other", "/other", "Ignored"),
            ]
            for filename, language, title, url, text in samples:
                path = site / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f'<html lang="{language}">' + article(f"<p>{text}</p>", title, url)
                                + "</html>", encoding="utf-8")
            (site / "index.html").write_text(article("Home", include="false", url="/"), encoding="utf-8")
            (site / "empty.html").write_text(article(""), encoding="utf-8")
            self.assertEqual(build_indexes(site), {"en": 2, "ja": 1, "ko": 1})
            english = json.loads((site / "search-en.json").read_text(encoding="utf-8"))
            self.assertEqual([doc["url"] for doc in english], ["/a", "/z"])
            japanese = (site / "search-ja.json").read_text(encoding="utf-8")
            self.assertIn("深層学習", japanese)
            self.assertNotIn('": ', japanese)
            outputs = {path.name: path.read_bytes() for path in site.glob("search-*.json")}
            build_indexes(site)
            self.assertEqual(outputs, {path.name: path.read_bytes() for path in site.glob("search-*.json")})

    def test_cli_writes_all_languages_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory)
            (site / "guide.html").write_text(article("<p>Guide text.</p>"), encoding="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("build_search.py")), str(site)],
                                    capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), "Search indexes: en=1, ja=0, ko=0 (1 total).")
            self.assertEqual(json.loads((site / "search-ja.json").read_text()), [])
            self.assertEqual(json.loads((site / "search-ko.json").read_text()), [])

    def test_missing_directory_and_metadata_fail_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory)
            with self.assertRaisesRegex(ValueError, "does not exist"):
                build_indexes(site / "missing")
            (site / "bad.html").write_text(article("Text", url=""), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bad.html.*root-relative URL"):
                build_indexes(site)


if __name__ == "__main__":
    unittest.main()
