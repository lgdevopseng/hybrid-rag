import json
import sqlite3
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DB_PATH


# Opens a connection to the KB database.
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Loads item records by id.
def fetch_items(ids: List[str]) -> List[Dict[str, Any]]:
    if not ids:
        return []
    conn = get_conn()
    q_marks = ",".join("?" for _ in ids)
    rows = conn.execute(f"SELECT id, json_data FROM items WHERE id IN ({q_marks})", ids).fetchall()
    conn.close()
    out = []
    for r in rows:
        data = json.loads(r["json_data"])
        out.append(data)
    return out

# Loads relation rows for the given source ids.
def fetch_relations_from(src_ids: List[str]) -> List[Tuple[str, str, str]]:
    if not src_ids:
        return []
    conn = get_conn()
    q_marks = ",".join("?" for _ in src_ids)
    rows = conn.execute(
        f"SELECT src_id, rel_type, dst_id FROM relations WHERE src_id IN ({q_marks})",
        src_ids,
    ).fetchall()
    conn.close()
    return [(r["src_id"], r["rel_type"], r["dst_id"]) for r in rows]
