#!/usr/bin/env python3
"""
Scraper for the-guided-meditation-site.com
Extracts free guided meditation scripts (text-based).
"""

import json
import os
import re
import time
import requests
from bs4 import BeautifulSoup, Tag

BASE_URL = "https://www.the-guided-meditation-site.com"
INDEX_URL = f"{BASE_URL}/guided-meditation-scripts.html"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "guided_meditation_site")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
}


def get_script_links() -> list[dict[str, str]]:
    """Get all meditation script links from the index page."""
    resp = requests.get(INDEX_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    links = []
    # Find all links that point to meditation script pages
    for a in soup.find_all("a", href=True):
        href = a["href"]
        title = a.get_text(strip=True)
        # Filter for meditation script pages (they typically have descriptive titles)
        if (href.endswith(".html") and
            href != "guided-meditation-scripts.html" and
            not href.startswith("http") and
            title and len(title) > 5):
            full_url = f"{BASE_URL}/{href}" if not href.startswith("/") else f"{BASE_URL}{href}"
            links.append({"title": title, "url": full_url})

    # Deduplicate by URL
    seen = set()
    unique = []
    for link in links:
        if link["url"] not in seen:
            seen.add(link["url"])
            unique.append(link)

    return unique


def scrape_script(url: str, title: str) -> dict[str, str | int] | None:
    """Scrape a single meditation script page."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ERROR fetching {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Try to find the main content area
    # Look for the article/content div
    content_div = soup.find("div", {"id": "content"}) or soup.find("article") or soup.find("div", class_=re.compile(r"content|article|post|entry"))

    if content_div is None:
        # Fallback: get all paragraphs from body
        content_div = soup.find("body")

    if content_div is None or not isinstance(content_div, Tag):
        print(f"  WARNING: no content found for {url}")
        return None

    # Extract text from paragraphs
    paragraphs = []
    for p in content_div.find_all(["p", "blockquote"]):
        text = p.get_text(strip=True)
        if text and len(text) > 10:
            paragraphs.append(text)

    if not paragraphs:
        print(f"  WARNING: no paragraphs found for {url}")
        return None

    full_text = "\n\n".join(paragraphs)

    return {
        "source": "the-guided-meditation-site.com",
        "title": title,
        "url": url,
        "text": full_text,
        "word_count": len(full_text.split()),
    }


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching script index...")
    links = get_script_links()
    print(f"Found {len(links)} script links")

    results = []
    for i, link in enumerate(links):
        print(f"[{i+1}/{len(links)}] Scraping: {link['title'][:60]}")
        script = scrape_script(link["url"], link["title"])
        if script:
            results.append(script)
            print(f"  OK: {script['word_count']} words")
        time.sleep(1.5)  # Be polite

    # Save all results
    output_path = os.path.join(OUTPUT_DIR, "scripts.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} scripts to {output_path}")

    # Also save individual text files for easy reading
    for script in results:
        slug = re.sub(r'[^a-z0-9]+', '_', script["title"].lower().strip())[:60]
        txt_path = os.path.join(OUTPUT_DIR, f"{slug}.txt")
        with open(txt_path, "w") as f:
            f.write(f"Title: {script['title']}\n")
            f.write(f"Source: {script['source']}\n")
            f.write(f"URL: {script['url']}\n")
            f.write(f"Words: {script['word_count']}\n")
            f.write("---\n\n")
            f.write(script["text"])

    return results


if __name__ == "__main__":
    main()
