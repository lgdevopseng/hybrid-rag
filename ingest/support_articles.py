from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import httpx

ARTICLE_ID_RE = re.compile(r"/articles/(\d+)-([^/?#]+)")
HEADING_RE = re.compile(r"^(#+)\s*(.*)$")

NOISE_LINES = {
    "english",
    "japanese",
    "welcome",
    "login",
    "sign up",
    "home",
    "solutions",
    "forums",
    "online help",
    "how can we help you today?",
    "enter your search term here...",
    "search",
    "new support ticket",
    "check ticket status",
    "solution home",
    "did you find it helpful? yes no",
    "send feedback",
    "sorry we couldn't be helpful. help us improve this article with your feedback.",
    "related articles",
    "print",
    "article views count",
    "image",
}

SUMMARY_HEADINGS = ("user case", "summary", "solution", "overview", "how to")

# Loads support article URLs from a text file.
def load_support_urls(path: Path) -> List[str]:
    if not path.exists():
        return []
    urls: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls

# Extracts article metadata from the URL path.
def extract_article_meta(url: str) -> Dict[str, str]:
    parsed = urlparse(url)
    match = ARTICLE_ID_RE.search(parsed.path)
    article_id = match.group(1) if match else parsed.path.rstrip("/").split("/")[-1]
    slug = match.group(2) if match else article_id
    source_locale = "ja" if "/ja/" in parsed.path else "en"
    return {
        "article_id": article_id,
        "slug": slug,
        "source_locale": source_locale,
    }

# Removes HTML tags while preserving readable line breaks.
def strip_tags_fragment(html_text: str) -> str:
    text = re.sub(r"(?is)<script.*?</script>", " ", html_text)
    text = re.sub(r"(?is)<style.*?</style>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|section|article|main|header|footer|li|tr|h[1-6])>", "\n", text)
    text = re.sub(r"(?i)<(p|div|section|article|main|header|footer|li|tr|h[1-6])[^>]*>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    lines = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split()).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)

# Gets the article title or falls back to a slug-based name.
def extract_title(html_text: str, fallback: str) -> str:
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", html_text)
    if not m:
        return fallback
    title = " ".join(html.unescape(m.group(1)).split()).strip()
    title = re.sub(r"\s*:\s*TestArchitect Support\s*$", "", title, flags=re.I)
    return title or fallback

# Normalizes raw article text into clean lines.
def normalize_lines(text: str) -> List[str]:
    lines: List[str] = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered in NOISE_LINES:
            continue
        lines.append(line)
    return lines

# Builds a short summary from the article lines.
def summarize_support_lines(lines: List[str]) -> str:
    if not lines:
        return ""

    summary: List[str] = []
    start_idx = 0
    for idx, line in enumerate(lines):
        if line.lower() in SUMMARY_HEADINGS:
            start_idx = idx
            break

    for line in lines[start_idx:]:
        lowered = line.lower()
        if lowered in NOISE_LINES:
            continue
        if lowered.startswith("more details can be found"):
            continue
        if lowered.startswith("did you find it helpful"):
            continue
        if lowered.startswith("option #") or lowered.startswith("step ") or lowered.startswith("tip:"):
            summary.append(line)
        elif line[:1].isupper() or line.endswith("."):
            summary.append(line)
        elif not summary:
            summary.append(line)
        if len(summary) >= 5:
            break

    if not summary:
        summary = lines[:5]

    text = " ".join(summary)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 520:
        text = text[:520].rsplit(" ", 1)[0] + "..."
    return text

# Fetches and normalizes one support article.
def fetch_support_article(url: str) -> Dict[str, Any]:
    resp = httpx.get(url, follow_redirects=True, timeout=30.0)
    resp.raise_for_status()
    html_text = resp.text
    meta = extract_article_meta(url)
    title = extract_title(html_text, meta["slug"].replace("-", " "))
    body_text = strip_tags_fragment(html_text)
    lines = normalize_lines(body_text)
    summary = summarize_support_lines(lines)
    raw_text = "\n".join(lines)

    return {
        "id": f"support.{meta['article_id']}",
        "kind": "support_article",
        "source_kind": "support_article",
        "source_url": url,
        "source_locale": meta["source_locale"],
        "article_id": meta["article_id"],
        "source_canonical_id": meta["article_id"],
        "slug": meta["slug"],
        "folder_category": "support",
        "name_en": title,
        "description_en": summary or title,
        "summary_en": summary or title,
        "raw_en": raw_text,
        "reference": title,
        "summary_title_en": title,
        "notes_en": [],
        "supported_platforms": [],
        "supported_os": [],
    }

# Ingests a list of support article URLs.
def ingest_support_articles(urls: List[str]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for url in urls:
        normalized = url.strip()
        if not normalized:
            continue
        try:
            record = fetch_support_article(normalized)
        except Exception as exc:
            print(f"[support ingest] skipped {normalized}: {exc}")
            continue
        key = ("support_article", record["id"])
        records[key] = record
    return records
