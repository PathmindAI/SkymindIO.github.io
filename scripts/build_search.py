#!/usr/bin/env python3
"""Build language-specific search indexes from the rendered wiki HTML."""

import argparse
from dataclasses import dataclass, field
from html.parser import HTMLParser
import json
from pathlib import Path


LANGUAGES = ("en", "ja", "ko")
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}
BLOCK_TAGS = HEADINGS | {
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl",
    "dt", "figcaption", "figure", "hr", "li", "ol", "p", "pre", "section",
    "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
}
EXCLUDED_TAGS = {"script", "style", "nav", "footer", "noscript", "template", "svg"}


@dataclass
class Element:
    tag: str
    attrs: dict
    children: list = field(default_factory=list)


def has_class(element, name):
    return name in (element.attrs.get("class") or "").split()


def excluded(element):
    return (element.tag in EXCLUDED_TAGS
            or has_class(element, "author-note")
            or "hidden" in element.attrs
            or element.attrs.get("aria-hidden") == "true")


class ArticleParser(HTMLParser):
    """Retain marked article trees, excluding the surrounding site layout."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.language = "en"
        self.articles = []
        self.stack = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "html":
            self.language = (attrs.get("lang") or "en").lower().split("-")[0]
        element = Element(tag, attrs)
        if not self.stack:
            if (tag == "article" and has_class(element, "wiki-article")
                    and attrs.get("data-search-include") == "true"):
                self.articles.append((self.language, element))
                self.stack.append(element)
            return
        self.stack[-1].children.append(element)
        if tag not in VOID_TAGS:
            self.stack.append(element)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        if self.stack:
            self.stack[-1].children.append(data)


def find_first(element, predicate):
    if excluded(element):
        return None
    if predicate(element):
        return element
    for child in element.children:
        if isinstance(child, Element):
            result = find_first(child, predicate)
            if result is not None:
                return result
    return None


def normalize(parts):
    return " ".join("".join(parts).split())


def plain_text(element):
    parts = []

    def visit(node):
        if isinstance(node, str):
            parts.append(node)
        elif not excluded(node):
            if node.tag in BLOCK_TAGS:
                parts.append(" ")
            for child in node.children:
                visit(child)
            if node.tag in BLOCK_TAGS:
                parts.append(" ")

    visit(element)
    return normalize(parts)


def heading_anchor(heading):
    if heading.attrs.get("id"):
        return heading.attrs["id"]
    anchor = find_first(heading, lambda node: node.tag == "a"
                        and (node.attrs.get("id") or node.attrs.get("name")))
    if anchor is not None:
        return anchor.attrs.get("id") or anchor.attrs["name"]
    return None


def extract_sections(body, page_url):
    sections = []
    parts = []
    title = ""
    section_url = page_url

    def finish():
        text = normalize(parts)
        if text:
            sections.append({"title": title, "text": text, "url": section_url})
        parts.clear()

    def visit(node):
        nonlocal title, section_url
        if isinstance(node, str):
            parts.append(node)
            return
        if excluded(node):
            return
        if node.tag in HEADINGS:
            finish()
            title = plain_text(node)
            anchor = heading_anchor(node)
            if anchor:
                section_url = page_url + "#" + anchor
            return
        if node.tag in BLOCK_TAGS:
            parts.append(" ")
        for child in node.children:
            visit(child)
        if node.tag in BLOCK_TAGS:
            parts.append(" ")

    visit(body)
    finish()
    return sections


def extract_document(article):
    body = find_first(article, lambda node: has_class(node, "article-body"))
    if body is None or not plain_text(body):
        return None
    title = normalize([article.attrs.get("data-search-title") or ""])
    url = article.attrs.get("data-search-url") or ""
    if not title or not url.startswith("/"):
        raise ValueError("Searchable articles need a title and a root-relative URL")
    header = find_first(article, lambda node: has_class(node, "article-header"))
    h1 = find_first(header, lambda node: node.tag == "h1") if header else None
    document = {"title": title, "url": url}
    if h1 is not None:
        document["fullTitle"] = plain_text(h1)
    document["sections"] = extract_sections(body, url)
    if not document["sections"]:
        return None
    return document


def build_indexes(site_directory):
    site_directory = Path(site_directory)
    if not site_directory.is_dir():
        raise ValueError("Rendered site directory does not exist: " + str(site_directory))
    indexes = {language: [] for language in LANGUAGES}
    for path in sorted(site_directory.rglob("*.html")):
        parser = ArticleParser()
        parser.feed(path.read_text(encoding="utf-8"))
        parser.close()
        for language, article in parser.articles:
            if language not in indexes:
                continue
            try:
                document = extract_document(article)
            except ValueError as error:
                raise ValueError(str(path) + ": " + str(error)) from error
            if document is not None:
                indexes[language].append(document)
    for language, documents in indexes.items():
        output = site_directory / ("search-" + language + ".json")
        output.write_text(json.dumps(documents, ensure_ascii=False,
                                     separators=(",", ":")) + "\n", encoding="utf-8")
    return {language: len(documents) for language, documents in indexes.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_directory", nargs="?", default="_site",
                        help="rendered Jekyll output directory (default: _site)")
    arguments = parser.parse_args()
    try:
        counts = build_indexes(arguments.site_directory)
    except (OSError, ValueError) as error:
        parser.exit(1, "Search indexing failed: " + str(error) + "\n")
    summary = ", ".join(language + "=" + str(counts[language]) for language in LANGUAGES)
    print("Search indexes: " + summary + " (" + str(sum(counts.values())) + " total).")


if __name__ == "__main__":
    main()
