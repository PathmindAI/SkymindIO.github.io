from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from check_search import check


class SearchCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'lesson.html').write_text('<html lang="en"><article data-search-include="true" data-search-url="/lesson"><h2 id="example">Example</h2></article></html>')
        self.entries = [{'title': 'Lesson', 'url': '/lesson', 'sections': [
            {'title': 'Example', 'text': 'An example passage.', 'url': '/lesson#example'}]}]
        for language in ('ja', 'ko'):
            (self.root / f'search-{language}.json').write_text('[]')

    def result(self):
        (self.root / 'search-en.json').write_text(json.dumps(self.entries))
        with redirect_stdout(StringIO()):
            return check(self.root)

    def test_valid_index(self):
        self.assertEqual(self.result(), 0)

    def test_missing_article_rejects_incomplete_index(self):
        self.entries = []
        self.assertEqual(self.result(), 1)

    def test_missing_fragment_rejects_dead_search_result(self):
        self.entries[0]['sections'][0]['url'] = '/lesson#absent'
        self.assertEqual(self.result(), 1)

    def test_empty_passage_rejected(self):
        self.entries[0]['sections'][0]['text'] = ' '
        self.assertEqual(self.result(), 1)

    def test_external_destination_rejected(self):
        self.entries[0]['sections'][0]['url'] = 'https://example.org/'
        self.assertEqual(self.result(), 1)


if __name__ == '__main__':
    unittest.main()
