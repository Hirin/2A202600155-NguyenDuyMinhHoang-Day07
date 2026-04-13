"""Weaviate collection migration: side-by-side approach.

Creates a new Harrier collection alongside the existing OpenAI collection,
without dropping the old one. Supports A/B benchmarking.

Usage:
    python scripts/migrate_weaviate.py --action create
    python scripts/migrate_weaviate.py --action status
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv(override=False)


# Collection naming convention
COLLECTIONS = {
    "openai_v1": {
        "name": "rag_tthc_openai_v1",
        "dim": 1536,
        "description": "text-embedding-3-small (1536-dim)",
    },
    "harrier_v1": {
        "name": "rag_tthc_harrier_v1",
        "dim": 1024,
        "description": "vietlegal-harrier-0.6b-Q8_0 (1024-dim)",
    },
}

# Expanded TTHC metadata properties for filtering
TTHC_PROPERTIES = [
    ("content", "TEXT"),
    ("doc_id", "TEXT"),
    ("ma_thu_tuc", "TEXT"),
    ("ten_thu_tuc", "TEXT"),
    ("linh_vuc", "TEXT"),
    ("cap_thuc_hien", "TEXT"),
    ("co_quan_thuc_hien", "TEXT"),
    ("agency_folder", "TEXT"),
    ("section_type", "TEXT"),
    ("section_heading", "TEXT"),
    ("parent_id", "TEXT"),
    ("chunk_type", "TEXT"),
    ("meta_json", "TEXT"),
]


def get_weaviate_client():
    """Connect to Weaviate cloud."""
    import weaviate
    from weaviate.classes.init import Auth

    url = os.getenv("WEAVIATE_URL", "")
    key = os.getenv("WEAVIATE_API_KEY", "")

    if not url or not key:
        print("❌ WEAVIATE_URL and WEAVIATE_API_KEY must be set in .env")
        sys.exit(1)

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(key),
    )
    return client


def create_harrier_collection(client, collection_name: str = "rag_tthc_harrier_v1"):
    """Create the Harrier collection with expanded TTHC metadata properties."""
    from weaviate.classes.config import Configure, Property, DataType

    if client.collections.exists(collection_name):
        print(f"⚠️  Collection '{collection_name}' already exists. Skipping.")
        return

    dt_map = {"TEXT": DataType.TEXT, "INT": DataType.INT}
    properties = [
        Property(name=name, data_type=dt_map[dt])
        for name, dt in TTHC_PROPERTIES
    ]

    client.collections.create(
        name=collection_name,
        properties=properties,
        vectorizer_config=Configure.Vectorizer.none(),
    )
    print(f"✅ Created collection '{collection_name}' with {len(properties)} properties")


def show_status(client):
    """Show status of all known collections."""
    print("\n=== Weaviate Collection Status ===\n")
    for key, info in COLLECTIONS.items():
        name = info["name"]
        exists = client.collections.exists(name)
        if exists:
            col = client.collections.get(name)
            agg = col.aggregate.over_all(total_count=True)
            count = agg.total_count or 0
            print(f"  ✅ {name}: {count:,} objects ({info['description']})")
        else:
            print(f"  ❌ {name}: does not exist ({info['description']})")
    print()


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Weaviate collection migration")
    ap.add_argument("--action", choices=["create", "status"], default="status")
    args = ap.parse_args()

    client = get_weaviate_client()

    try:
        if args.action == "create":
            create_harrier_collection(client)
        show_status(client)
    finally:
        client.close()


if __name__ == "__main__":
    main()
