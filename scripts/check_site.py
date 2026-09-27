#!/usr/bin/env python3
"""Check links in a generated Jekyll site, using only the Python standard library.

Usage: python3 scripts/check_site.py _site

Checks anchor links throughout English HTML pages, including home, about, shared
navigation, and redirect pages. Japanese and Korean pages under /jp and /kr are
excluded as sources, but remain valid destinations. External URLs, non-HTTP
schemes, media resources, and JavaScript-generated navigation are not audited.
HTML meta-refresh redirects are followed to their final page and fragment.
The 404 fallback never satisfies a missing destination.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote, urljoin, urlsplit


@dataclass(frozen=True)
class Link:
    href: str
    line: int


class Document(HTMLParser):
    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.anchors: set[str] = set()
        self.links: list[Link] = []
        self.redirect: Link | None = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id") is not None:
            self.anchors.add(attrs["id"])
        if tag == "a" and attrs.get("name") is not None:
            self.anchors.add(attrs["name"])
        if tag in {"a", "area"} and attrs.get("href") is not None:
            self.links.append(Link(attrs["href"], self.getpos()[0]))
        if tag == "meta" and attrs.get("http-equiv", "").lower() == "refresh":
            match = re.search(r";\s*url\s*=\s*(.*)$", attrs.get("content", ""), re.I)
            if match:
                self.redirect = Link(match[1].strip().strip("\"'"), self.getpos()[0])


@dataclass
class Report:
    pages: int = 0
    links: int = 0
    issues: list[str] = field(default_factory=list)


class SiteChecker:
    def __init__(self, root: Path, site_url: str = "https://wiki.pathmind.com"):
        self.root = Path(root).resolve()
        self.site_url = site_url.rstrip("/") + "/"
        self.host = urlsplit(self.site_url).netloc.lower()
        self.documents: dict[Path, Document] = {}

    def document(self, path: Path) -> Document:
        if path not in self.documents:
            self.documents[path] = Document(path.read_text(encoding="utf-8"))
        return self.documents[path]

    def page_url(self, path: Path) -> str:
        relative = path.relative_to(self.root).as_posix()
        if path.name == "index.html":
            relative = relative[:-len("index.html")]
        return urljoin(self.site_url, relative)

    def local(self, url: str) -> bool:
        parts = urlsplit(url)
        return parts.scheme in {"http", "https"} and parts.netloc.lower() == self.host

    def resolve(self, path: str) -> Path | None:
        path = unquote(path).lstrip("/")
        candidates = (self.root / path, self.root / (path.rstrip("/") + ".html"),
                      self.root / path / "index.html")
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.is_relative_to(self.root) and candidate.is_file():
                return candidate
        return None

    def destination_error(self, url: str) -> str | None:
        visited: set[tuple[Path, str]] = set()
        for _ in range(32):
            if not self.local(url):
                return None
            parts = urlsplit(url)
            target = self.resolve(parts.path)
            if target is None:
                return f"missing page {unquote(parts.path) or '/'}"
            if target == self.root / "404.html":
                return "destination is the 404 fallback"
            if target.suffix.lower() not in {".html", ".htm"}:
                return None
            state = (target, parts.fragment)
            if state in visited:
                return f"redirect cycle at {parts.path}"
            visited.add(state)
            document = self.document(target)
            if document.redirect is not None:
                url = urljoin(url, document.redirect.href)
                continue
            # Browser text fragments can accompany an ID or appear on their own.
            fragment = unquote(parts.fragment.split(":~:text=", 1)[0])
            if fragment and fragment not in document.anchors:
                return f"missing fragment #{fragment} in {target.relative_to(self.root)}"
            return None
        return "redirect chain exceeds 32 pages"

    def check(self) -> Report:
        report = Report()
        if not (self.root / "index.html").is_file():
            report.issues.append("index.html: missing home page")
        pages = sorted(path for path in self.root.rglob("*.html")
                       if path.relative_to(self.root).parts[0] not in {"jp", "kr"}
                       and path != self.root / "404.html")
        for page in pages:
            report.pages += 1
            document = self.document(page)
            base = self.page_url(page)
            links = list(document.links)
            if document.redirect is not None:
                # Validate redirect pages even when nothing links to them yet.
                links.append(document.redirect)
            for link in links:
                url = urljoin(base, link.href)
                if not self.local(url):
                    continue
                report.links += 1
                error = self.destination_error(url)
                if error:
                    report.issues.append(
                        f"{page.relative_to(self.root)}:{link.line}: {link.href!r}: {error}")
        return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_dir", type=Path, help="Jekyll output directory")
    parser.add_argument("--site-url", default="https://wiki.pathmind.com",
                        help="Public origin used to identify same-site absolute links")
    args = parser.parse_args(argv)
    if not args.site_dir.is_dir():
        parser.error(f"Generated site directory does not exist: {args.site_dir}")
    if urlsplit(args.site_url).scheme not in {"http", "https"} or not urlsplit(args.site_url).netloc:
        parser.error("--site-url must be an HTTP(S) origin")
    report = SiteChecker(args.site_dir, args.site_url).check()
    for issue in report.issues[:50]:
        print(issue)
    if len(report.issues) > 50:
        print(f"... {len(report.issues) - 50} additional failures")
    print(f"Checked {report.pages} English/shared HTML pages and {report.links} internal links; "
          f"{len(report.issues)} failures.")
    return 1 if report.issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
