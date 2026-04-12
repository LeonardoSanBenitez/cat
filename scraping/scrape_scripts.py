#!/usr/bin/env python3
"""
Targeted scraper for meditation scripts from multiple free sources.
Only fetches actual meditation instruction texts.
"""

import json
import os
import re
import time
import requests
from bs4 import BeautifulSoup, Tag

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
}

# ── Known meditation script URLs ──────────────────────────────────────

GUIDED_MEDITATION_SITE_SCRIPTS = [
    ("The Forest Speaks", "https://www.the-guided-meditation-site.com/guided-meditation-script-forest-speaks.html"),
    ("Rainbow Meditation and Pot of Gold Visualization", "https://www.the-guided-meditation-site.com/guided-meditation-script-rainbow.html"),
    ("Peace, Tranquility and Healing", "https://www.the-guided-meditation-site.com/guided-meditation-script-peace-tranquility-healing.html"),
    ("Buddhist Guided Meditation", "https://www.the-guided-meditation-site.com/buddhist-guided-meditation-script.html"),
    ("Body Awareness Meditation", "https://www.the-guided-meditation-site.com/body-awareness-meditation.html"),
    ("The Kingfisher", "https://www.the-guided-meditation-site.com/the-kingfisher.html"),
    ("Ribbons of Healing Light", "https://www.the-guided-meditation-site.com/ribbons-of-healing-light.html"),
    ("Letting Go of Work/Home Stresses", "https://www.the-guided-meditation-site.com/letting-go-of-workhome-stresses-25-mins.html"),
    ("Radiant Being", "https://www.the-guided-meditation-site.com/radiant-being.html"),
    ("Angel Healing Meditation", "https://www.the-guided-meditation-site.com/angel-healing-meditation-for-self-and-others-to-send-healing-to.html"),
    ("Gently Down The Stream", "https://www.the-guided-meditation-site.com/gently-down-the-stream.html"),
    ("Bluebell Wood Meditation", "https://www.the-guided-meditation-site.com/bluebell-wood-meditation-script.html"),
    ("Forest Waterfall", "https://www.the-guided-meditation-site.com/forest-waterfall.html"),
    ("Trust Walk - Path to Freedom and Self Love", "https://www.the-guided-meditation-site.com/trust-walk-a-path-to-freedom-and-self-love.html"),
    ("It's Like This - Mindfulness Meditation", "https://www.the-guided-meditation-site.com/its-like-this-mindfulness-meditation.html"),
    ("Healing Guided Meditation", "https://www.the-guided-meditation-site.com/healing-guided-meditation-script.html"),
    ("The Hammock", "https://www.the-guided-meditation-site.com/the-hammock.html"),
    ("Heart Breath", "https://www.the-guided-meditation-site.com/heart-breath.html"),
    ("Guided Loving-Kindness - Buddhist Meditation", "https://www.the-guided-meditation-site.com/buddhist-meditation-script-guided-lovingkindness.html"),
    ("Acknowledge All Awareness", "https://www.the-guided-meditation-site.com/acknowledge-all-awareness.html"),
    ("Coastal Path Meditation", "https://www.the-guided-meditation-site.com/coastal-path-meditation.html"),
    ("Stillness and Centered Meditation", "https://www.the-guided-meditation-site.com/stillness-and-centered-meditation.html"),
    ("The Ocean", "https://www.the-guided-meditation-site.com/the-ocean.html"),
    ("Flowing Stream Meditation", "https://www.the-guided-meditation-site.com/flowing-stream-meditation.html"),
    ("Under Waterfall", "https://www.the-guided-meditation-site.com/under-waterfall.html"),
    ("Bliss For Your Brain", "https://www.the-guided-meditation-site.com/bliss-for-your-brain.html"),
]


def scrape_page(url: str) -> str | None:
    """Fetch a page and extract the main text content."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ERROR: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove navigation, header, footer, sidebar elements
    for tag in soup.find_all(["nav", "header", "footer", "aside"]):
        tag.decompose()
    for tag in soup.find_all(class_=re.compile(r"nav|menu|sidebar|footer|header|widget|comment|share|social")):
        tag.decompose()

    # Try to find the main content
    content = (soup.find("article") or
               soup.find("div", {"id": "content"}) or
               soup.find("div", class_=re.compile(r"^(content|article|post|entry)")) or
               soup.find("main") or
               soup.find("body"))

    if not content or not isinstance(content, Tag):
        return None

    # Extract paragraphs
    paragraphs = []
    for p in content.find_all(["p", "blockquote"]):
        text = p.get_text(strip=True)
        # Filter out very short paragraphs and navigation-like text
        if (text and len(text) > 20 and
            not text.startswith("Copyright") and
            not text.startswith("All rights") and
            "cookie" not in text.lower() and
            "subscribe" not in text.lower()):
            paragraphs.append(text)

    if not paragraphs:
        return None

    return "\n\n".join(paragraphs)


def scrape_source(scripts: list[tuple[str, str]], source_name: str) -> list[dict[str, str | int]]:
    """Scrape a list of (title, url) pairs from a source."""
    results = []
    for i, (title, url) in enumerate(scripts):
        print(f"  [{i+1}/{len(scripts)}] {title[:60]}")
        text = scrape_page(url)
        if text and len(text.split()) > 30:
            results.append({
                "source": source_name,
                "title": title,
                "url": url,
                "text": text,
                "word_count": len(text.split()),
            })
            print(f"    OK: {len(text.split())} words")
        else:
            print(f"    SKIP: insufficient content")
        time.sleep(1.0)
    return results


def save_results(results: list[dict[str, str | int]], filename: str) -> str:
    """Save scraped results to JSON and individual text files."""
    out_dir = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(out_dir, exist_ok=True)

    # Save combined JSON
    json_path = os.path.join(out_dir, "scripts.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save individual text files
    for script in results:
        slug = re.sub(r'[^a-z0-9]+', '_', script["title"].lower().strip())[:60]
        txt_path = os.path.join(out_dir, f"{slug}.txt")
        with open(txt_path, "w") as f:
            f.write(f"Title: {script['title']}\n")
            f.write(f"Source: {script['source']}\n")
            f.write(f"URL: {script['url']}\n")
            f.write(f"Words: {script['word_count']}\n")
            f.write("---\n\n")
            f.write(script["text"])

    print(f"  Saved {len(results)} scripts to {json_path}")
    return json_path


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results: list[dict[str, str | int]] = []

    # Source 1: the-guided-meditation-site.com
    print("\n=== Source: the-guided-meditation-site.com ===")
    results = scrape_source(GUIDED_MEDITATION_SITE_SCRIPTS, "the-guided-meditation-site.com")
    if results:
        save_results(results, "guided_meditation_site")
        all_results.extend(results)

    # Save combined dataset
    combined_path = os.path.join(OUTPUT_DIR, "all_scripts.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n=== TOTAL: {len(all_results)} meditation scripts collected ===")
    print(f"Combined dataset: {combined_path}")


if __name__ == "__main__":
    main()
