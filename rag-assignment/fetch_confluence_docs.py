import os
import re
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

CONFLUENCE_EMAIL = os.getenv("ATLASSIAN_EMAIL")
CONFLUENCE_API_TOKEN = os.getenv("ATLASSIAN_API_KEY")
BASE_URL = "https://repairtechsolutions.atlassian.net/wiki/api/v2"

SPACE_KEYS = ["RSKB", "SKB"]

OUTPUT_DIR = Path(__file__).parent / "fetched-confluence-docs"


def get_auth():
    if not CONFLUENCE_EMAIL or not CONFLUENCE_API_TOKEN:
        raise ValueError(
            "Missing env vars! Set ATLASSIAN_EMAIL and ATLASSIAN_API_KEY"
        )
    return HTTPBasicAuth(CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN)


def get_space_id_by_key(space_key: str) -> str:
    url = f"{BASE_URL}/spaces"
    params = {"keys": space_key}
    response = requests.get(url, auth=get_auth(), params=params)

    if response.status_code != 200:
        print(f"Error fetching space: {response.status_code}")
        print(response.text)
        return None

    results = response.json().get("results", [])
    if results:
        return results[0].get("id")
    return None


def get_pages_in_space(space_id: str, limit: int = 250) -> list:
    all_pages = []
    cursor = None

    while True:
        url = f"{BASE_URL}/spaces/{space_id}/pages"
        params = {"limit": limit, "body-format": "storage"}
        if cursor:
            params["cursor"] = cursor

        response = requests.get(url, auth=get_auth(), params=params)

        if response.status_code != 200:
            print(f"Error fetching pages: {response.status_code}")
            print(response.text)
            break

        data = response.json()
        pages = data.get("results", [])
        all_pages.extend(pages)

        print(f"  Fetched {len(pages)} pages (total: {len(all_pages)})")

        next_link = data.get("_links", {}).get("next")
        if not next_link:
            break

        if "cursor=" in next_link:
            cursor = next_link.split("cursor=")[1].split("&")[0]
        else:
            break

    return all_pages


def clean_filename(title: str) -> str:
    clean = re.sub(r'[<>:"/\\|?*]', '', title)
    clean = re.sub(r'\s+', '_', clean)
    return clean[:100]


def save_page_to_file(page: dict, output_dir: Path):
    page_id = page.get("id")
    title = page.get("title", "Untitled")

    body = ""
    if "body" in page:
        body_obj = page["body"]
        if "storage" in body_obj:
            body = body_obj["storage"].get("value", "")
        elif "atlas_doc_format" in body_obj:
            body = body_obj["atlas_doc_format"].get("value", "")
        elif isinstance(body_obj, str):
            body = body_obj

    filename = f"{clean_filename(title)}.md"
    filepath = output_dir / filename

    content = f"""---
id: "{page_id}"
title: "{title}"
space_id: "{page.get('spaceId', '')}"
status: "{page.get('status', '')}"
created_at: "{page.get('createdAt', '')}"
---

# {title}

{body}
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  Saved: {filename}")


def list_available_spaces():
    print(f"\n🔗 Using API: {BASE_URL}")
    print("\n📋 Spaces you have access to:")
    url = f"{BASE_URL}/spaces"
    params = {"limit": 50}
    response = requests.get(url, auth=get_auth(), params=params)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return []

    spaces = response.json().get("results", [])
    for space in spaces:
        print(f"  - {space.get('key')}: {space.get('name')} (ID: {space.get('id')})")

    return spaces


def main():
    print("🚀 Confluence Document Extractor")
    print("=" * 50)

    list_available_spaces()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_saved = 0

    for space_key in SPACE_KEYS:
        print(f"\n📁 Processing space: {space_key}")

        space_id = get_space_id_by_key(space_key)
        if not space_id:
            print(f"  ❌ Could not find space or no access. Skipping.")
            continue

        print(f"  Found space ID: {space_id}")
        space_dir = OUTPUT_DIR / space_key
        space_dir.mkdir(exist_ok=True)

        pages = get_pages_in_space(space_id)
        print(f"  Found {len(pages)} pages")

        for page in pages:
            if page.get("status") == "archived":
                continue
            save_page_to_file(page, space_dir)
            total_saved += 1

    print("\n" + "=" * 50)
    print(f"✅ Done! Saved {total_saved} documents to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
