import json
import math
import re
import sys
import sqlite3
from pathlib import Path
from typing import List, Dict, Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL_EMBED
from rag.db import get_conn, fetch_items, fetch_relations_from
from rag.api_usage import log_api_usage

client = OpenAI(api_key=OPENAI_API_KEY)

FTS_TOKEN_RE = re.compile(r"[0-9A-Za-z_]+|[\u3040-\u30ff\u4e00-\u9fff]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "me", "of", "on", "or",
    "please", "setting", "settings", "the", "to", "use", "using", "what",
    "when", "where", "which", "with", "would", "action", "actions", "built",
    "builtin", "built-in", "tell", "show", "explain",
}

# Builds an embedding for a query string.
def embed_query(query: str) -> list[float]:
    resp = client.embeddings.create(
        model=OPENAI_MODEL_EMBED,
        input=[query],
        encoding_format="float",
    )
    usage = getattr(resp, "usage", None)
    if usage:
        log_api_usage("embed_query", usage, model=OPENAI_MODEL_EMBED)
    return resp.data[0].embedding

# Computes cosine similarity between two vectors.
def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

# Builds an FTS query from the raw text.
def build_fts_query(query: str) -> str:
    tokens = FTS_TOKEN_RE.findall(query.lower())
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"' for token in tokens)

# Builds a focus query without common stopwords.
def build_focus_query(query: str) -> str:
    tokens = [t for t in FTS_TOKEN_RE.findall(query.lower()) if t not in STOPWORDS]
    if not tokens:
        return ""
    return " ".join(tokens).strip()

# Finds exact title or slug matches in the KB.
def find_exact_match_ids(conn, query: str) -> List[str]:
    q = query.strip().lower()
    if not q:
        return []

    focus_q = build_focus_query(query)
    rows = conn.execute(
        """
        SELECT id, kind, name_en, slug
        FROM items
        WHERE lower(trim(name_en)) = ?
           OR lower(trim(slug)) = ?
        """,
        (q, q),
    ).fetchall()
    if not rows and focus_q and focus_q != q:
        rows = conn.execute(
            """
            SELECT id, kind, name_en, slug
            FROM items
            WHERE lower(trim(name_en)) = ?
               OR lower(trim(slug)) = ?
            """,
            (focus_q, focus_q.replace(" ", "-")),
        ).fetchall()
    action_ids = [r["id"] for r in rows if r["kind"] == "action"]
    setting_ids = [r["id"] for r in rows if r["kind"] != "action"]
    return action_ids + setting_ids

# Looks up one item id by its canonical name.
def _lookup_item_id_by_name(conn, name: str) -> List[str]:
    row = conn.execute(
        """
        SELECT id
        FROM items
        WHERE lower(trim(name_en)) = ?
           OR lower(trim(slug)) = ?
        LIMIT 1
        """,
        (name, name.replace(" ", "-")),
    ).fetchone()
    return [row["id"]] if row else []

# Infers likely action or setting ids from query intent.
def find_intent_ids(conn, query: str) -> List[str]:
    text = query.lower()
    ids: List[str] = []

    def has_word(term: str) -> bool:
        return re.search(rf"\b{re.escape(term)}\b", text) is not None

    def add_name(name: str):
        for item_id in _lookup_item_id_by_name(conn, name):
            if item_id not in ids:
                ids.append(item_id)

    if any(term in text for term in ("select", "choose", "pick")) and any(
        term in text for term in ("combo box", "combobox", "drop down", "dropdown", "list box", "listbox", "list control", "list item", "item", "control", "list")
    ):
        add_name("select")

    if "select" in text and "item" in text and any(term in text for term in ("list", "control", "combo", "dropdown", "drop down")):
        add_name("select")

    if any(term in text for term in ("selected", "selection", "checked")) and any(
        term in text for term in ("combo box", "combobox", "list box", "listbox", "list view", "list", "control", "item")
    ):
        add_name("check selected items")
        add_name("check selected count")
        add_name("check list item order")

    if any(term in text for term in ("文字列", "テキスト")) and any(
        term in text for term in ("存在", "含む", "含ま", "一致", "比較", "確認", "チェック")
    ):
        add_name("check text contains")
        add_name("check text exists")

    if any(term in text for term in ("contains", "contain", "exists", "exist", "present", "presence")) and any(
        term in text for term in ("text", "string", "substring", "value")
    ):
        add_name("check text contains")

    value_match_terms = ("verify", "validate", "confirm", "match", "matched", "matching", "recorded", "expected", "compare")
    if any(term in text for term in value_match_terms) and "value" in text:
        add_name("check value")
        add_name("check text contains")
        add_name("check text exists")
        if any(term in text for term in ("table cell", "cell", "table", "row", "column", "grid")):
            add_name("check table cell value")

    if "check value" in text:
        add_name("check value")

    if "check table cell value" in text:
        add_name("check table cell value")

    if any(term in text for term in ("text box", "textbox", "text field", "textfield")) or any(
        has_word(term) for term in ("input", "type", "enter", "fill")
    ):
        if any(has_word(term) for term in ("input", "type", "enter", "fill")):
            add_name("enter")

    if "click" in text and any(term in text for term in ("button", "window", "control", "element", "item")):
        add_name("click")

    return ids

# Ranks KB items using embeddings, FTS, and exact matches.
def hybrid_search(query: str, k: int = 5) -> List[str]:
    conn = get_conn()
    try:
        exact_ids = find_exact_match_ids(conn, query)
        intent_ids = find_intent_ids(conn, query)
        fts_query = build_fts_query(query)
        fts_scores = {}
        if fts_query:
            fts_rows = conn.execute("""
                SELECT id, bm25(items_fts) AS rank
                FROM items_fts
                WHERE items_fts MATCH ?
                ORDER BY rank
                LIMIT 20
            """, (fts_query,)).fetchall()
            fts_scores = {r["id"]: -r["rank"] for r in fts_rows}  # lower bm25 = better

        # Embedding similarity
        q_emb = embed_query(query)
        emb_rows = conn.execute("SELECT id, embedding FROM embeddings").fetchall()
        emb_scores = {}
        for r in emb_rows:
            emb = json.loads(r["embedding"])
            emb_scores[r["id"]] = cosine_sim(q_emb, emb)

        # Combine
        scores = {}
        for item_id, s in emb_scores.items():
            scores[item_id] = 0.7 * s + 0.3 * fts_scores.get(item_id, 0.0)

        for item_id in exact_ids:
            scores[item_id] = scores.get(item_id, 0.0) + 100.0

        for item_id in intent_ids:
            scores[item_id] = scores.get(item_id, 0.0) + 60.0

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [item_id for item_id, _ in ranked[:k]]
    except sqlite3.OperationalError:
        # If FTS5 still rejects the query shape, fall back to embeddings only.
        q_emb = embed_query(query)
        emb_rows = conn.execute("SELECT id, embedding FROM embeddings").fetchall()
        emb_scores = {}
        for r in emb_rows:
            emb = json.loads(r["embedding"])
            emb_scores[r["id"]] = cosine_sim(q_emb, emb)
        ranked = sorted(emb_scores.items(), key=lambda kv: kv[1], reverse=True)
        return [item_id for item_id, _ in ranked[:k]]
    finally:
        conn.close()

# Adds a small set of related items from graph relations.
def expand_with_relations(seed_ids: List[str], max_extra: int = 2) -> List[str]:
    rels = fetch_relations_from(seed_ids)
    extra: List[str] = []
    for _, _rel, dst_id in rels:
        if dst_id not in seed_ids and dst_id not in extra:
            extra.append(dst_id)
            if len(extra) >= max_extra:
                break
    return seed_ids + extra

# Retrieves full KB records for the query.
def retrieve_for_query(query: str, k: int = 5, use_graph: bool = True) -> List[Dict[str, Any]]:
    base_ids = hybrid_search(query, k=k)
    exact_ids: List[str] = []
    conn = None
    try:
        conn = get_conn()
        exact_ids = find_exact_match_ids(conn, query)
    finally:
        if conn is not None:
            conn.close()
    if use_graph:
        all_ids = expand_with_relations(base_ids)
    else:
        all_ids = base_ids
    items = fetch_items(all_ids)
    exact_set = set(exact_ids)
    for item in items:
        item["exact_match"] = item.get("id") in exact_set
    items.sort(key=lambda item: (not item.get("exact_match", False), item.get("id", "")))
    return items
