"""Embed the curated casting-defect knowledge base into a local ChromaDB
collection, persisted to disk, for the Root-Cause Agent to query.

    python -m src.knowledge.ingest

Re-running is safe -- entries are upserted by defect_type-derived id, so
re-ingesting after editing data/knowledge_base/casting_defects.json
updates existing entries rather than duplicating them.
"""

import json
import re
from pathlib import Path

from src.knowledge import chroma_compat  # noqa: F401  (must precede `import chromadb`)

import chromadb

REPO = Path(__file__).resolve().parents[2]
KNOWLEDGE_BASE_PATH = REPO / "data/knowledge_base/casting_defects.json"
CHROMA_DB_DIR = REPO / "data/chroma_db"
COLLECTION_NAME = "casting_defects"


def _slugify(defect_type: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", defect_type.lower()).strip("_")


def load_entries(path: Path = KNOWLEDGE_BASE_PATH) -> list:
    with open(path) as f:
        return json.load(f)["entries"]


def ingest(path: Path = KNOWLEDGE_BASE_PATH, persist_dir: Path = CHROMA_DB_DIR) -> chromadb.api.models.Collection.Collection:
    """Embed and upsert every knowledge-base entry into the persisted collection.

    Uses chromadb's default embedding function (all-MiniLM-L6-v2, run
    locally) -- no separate embedding API needed at this scale (9 entries).
    """
    entries = load_entries(path)

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    collection.upsert(
        ids=[_slugify(e["defect_type"]) for e in entries],
        documents=[f"{e['defect_type']}: {e['process_causes']}" for e in entries],
        metadatas=[{"defect_type": e["defect_type"], "process_causes": e["process_causes"]} for e in entries],
    )
    return collection


def main():
    collection = ingest()
    print(f"Ingested {collection.count()} entries into '{COLLECTION_NAME}' at {CHROMA_DB_DIR}")


if __name__ == "__main__":
    main()
