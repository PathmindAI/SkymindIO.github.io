"""Regression fixtures for the publish-blocking local link check."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from check_site import SiteChecker, main


class SiteCheckerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, path, html):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")

    def report(self):
        return SiteChecker(self.root).check()

    def test_valid_relative_encoded_named_and_nested_links(self):
        self.write("index.html", '<a href="/about/">About</a><a href="guide%20one#caf%C3%A9">Guide</a>'
                   '<a href="/nested/page.html#legacy">Nested</a><a href="/nested/page#legacy">Alias</a>')
        self.write("about/index.html", '<a href="../guide%20one#caf%C3%A9">Relative</a><a href="/">Home</a>')
        self.write("guide one.html", '<h2 id="café">Guide</h2>')
        self.write("nested/page.html", '<a name="legacy"></a>')
        self.assertEqual(self.report().issues, [])

    def test_missing_page_fails_even_with_a_working_404(self):
        self.write("index.html", '<a href="/missing">Missing</a>')
        self.write("404.html", '<meta http-equiv="refresh" content="0; url=/">')
        self.assertIn("missing page /missing", self.report().issues[0])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.root)]), 1)

    def test_missing_fragment_fails_on_an_existing_page(self):
        self.write("index.html", '<a href="/guide#absent">Missing section</a>')
        self.write("guide.html", '<h2 id="present">Present</h2>')
        self.assertIn("missing fragment #absent", self.report().issues[0])

    def test_redirect_chain_follows_target_fragment(self):
        self.write("index.html", '<a href="/old#obsolete">Old URL</a>')
        self.write("old.html", '<meta http-equiv="refresh" content="0; url=/middle">')
        self.write("middle.html", '<meta http-equiv="refresh" content="0; url=/guide#section">')
        self.write("guide.html", '<h2 id="section">Destination</h2>')
        self.assertEqual(self.report().issues, [])
        self.write("guide.html", '<h2 id="wrong">Destination changed</h2>')
        self.assertTrue(any("missing fragment #section" in issue for issue in self.report().issues))

    def test_redirect_cycles_fail(self):
        self.write("index.html", '<a href="/a">Cycle</a>')
        self.write("a.html", '<meta http-equiv="refresh" content="0; url=/b">')
        self.write("b.html", '<meta http-equiv="refresh" content="0; url=/a">')
        self.assertTrue(any("redirect cycle" in issue for issue in self.report().issues))

    def test_explicit_404_and_redirect_to_404_fail(self):
        self.write("index.html", '<a href="/404.html">Fallback</a><a href="/old">Old</a>')
        self.write("old.html", '<meta http-equiv="refresh" content="0; url=/404.html">')
        self.write("404.html", '<meta http-equiv="refresh" content="0; url=/">')
        self.assertTrue(all("404 fallback" in issue for issue in self.report().issues))
        self.assertGreaterEqual(len(self.report().issues), 2)

    def test_external_links_and_translated_bodies_are_out_of_scope(self):
        self.write("index.html", '<a href="https://example.com/missing">External</a>'
                   '<a href="mailto:hello@example.com">Email</a><a href="/jp/guide#ok">Japanese</a>')
        self.write("jp/guide.html", '<h2 id="ok"></h2><a href="/unreviewed">Not checked yet</a>')
        self.write("kr/guide.html", '<a href="/also-unreviewed">Not checked yet</a>')
        report = self.report()
        self.assertEqual(report.issues, [])
        self.assertEqual(report.pages, 1)

    def test_same_site_absolute_links_are_checked(self):
        self.write("index.html", '<a href="http://wiki.pathmind.com/missing">Local</a>')
        self.assertIn("missing page /missing", self.report().issues[0])

    def test_empty_site_cannot_pass(self):
        self.assertIn("missing home page", self.report().issues[0])


if __name__ == "__main__":
    unittest.main()
