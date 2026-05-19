import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from openai import OpenAI
from config import DOCS_ROOT, DB_PATH, OPENAI_API_KEY, OPENAI_MODEL_EMBED, SUPPORT_URLS_FILE, SUPPORT_URLS, SUPPORT_ONLY_BUILD
from ingest.parser import ingest_all
from ingest.support_articles import ingest_support_articles, load_support_urls
from rag.api_usage import log_api_usage

client = OpenAI(api_key=OPENAI_API_KEY)

# Gets an embedding vector for a KB record.
def get_embedding(text: str) -> list[float]:
    text = text.replace("\n", " ")
    resp = client.embeddings.create(
        model=OPENAI_MODEL_EMBED,
        input=[text],
        encoding_format="float",
    )
    usage = getattr(resp, "usage", None)
    if usage:
        log_api_usage("get_embedding", usage, model=OPENAI_MODEL_EMBED)
    return resp.data[0].embedding

# Creates the SQLite tables used by the KB.
def ensure_schema(conn: sqlite3.Connection):
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS items (
      id TEXT PRIMARY KEY,
      kind TEXT,
      slug TEXT,
      folder_category TEXT,
      name_en TEXT,
      name_ja TEXT,
      description_en TEXT,
      description_ja TEXT,
      json_data TEXT
    )
    """)
    c.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS items_fts
    USING fts5(id, name_en, name_ja, description_en, description_ja, raw_en, raw_ja)
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
      id TEXT PRIMARY KEY,
      embedding BLOB
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS relations (
      src_id TEXT,
      rel_type TEXT,
      dst_id TEXT
    )
    """)
    conn.commit()

# Builds and stores the knowledge base.
def build_kb():
    records: Dict[Tuple[str, str], Dict[str, Any]] = {}

    if not SUPPORT_ONLY_BUILD:
        records.update(ingest_all(Path(DOCS_ROOT)))

    support_urls = []
    support_urls.extend(load_support_urls(Path(SUPPORT_URLS_FILE)))
    if SUPPORT_URLS.strip():
        for raw_url in SUPPORT_URLS.replace("\r", "\n").replace(",", "\n").splitlines():
            url = raw_url.strip()
            if url:
                support_urls.append(url)
    support_urls = list(dict.fromkeys(support_urls))
    if support_urls:
        records.update(ingest_support_articles(support_urls))

    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)
    c = conn.cursor()

    for (_kind, _slug), rec in records.items():
        item_id = rec["id"]
        print("Processing", item_id)
        search_text = f"{rec.get('kind')}: {rec.get('name_en') or ''}. {rec.get('description_en') or rec.get('summary_en') or ''}"
        emb = get_embedding(search_text)

        json_data = json.dumps(rec, ensure_ascii=False)

        c.execute("""
        INSERT OR REPLACE INTO items
          (id, kind, slug, folder_category, name_en, name_ja, description_en, description_ja, json_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item_id,
            rec.get("kind"),
            rec.get("slug"),
            rec.get("folder_category"),
            rec.get("name_en"),
            rec.get("name_ja"),
            rec.get("description_en"),
            rec.get("description_ja"),
            json_data,
        ))

        c.execute("""
        INSERT OR REPLACE INTO items_fts
          (id, name_en, name_ja, description_en, description_ja, raw_en, raw_ja)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            item_id,
            rec.get("name_en") or "",
            rec.get("name_ja") or "",
            rec.get("description_en") or "",
            rec.get("description_ja") or "",
            rec.get("raw_en") or "",
            rec.get("raw_ja") or "",
        ))

        c.execute("""
        INSERT OR REPLACE INTO embeddings (id, embedding)
        VALUES (?, ?)
        """, (item_id, json.dumps(emb)))

        # relations: action -> setting
        if rec.get("kind") == "action":
            for sid in rec.get("applicable_setting_ids") or []:
                c.execute("""
                INSERT INTO relations (src_id, rel_type, dst_id)
                VALUES (?, 'USES_SETTING', ?)
                """, (item_id, sid))

    conn.commit()
    conn.close()
    print("KB built at", DB_PATH)


if __name__ == "__main__":
    build_kb()
