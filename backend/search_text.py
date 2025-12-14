# backend/search_text.py

import os
import sqlite3
from typing import List, Dict

# ----------------------------------------------------------
# DB 설정 (met20k/metadata.db)
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "met20k", "metadata.db")


# ----------------------------------------------------------
# LIKE 기반 텍스트 검색
# ----------------------------------------------------------
def search_text(query: str, limit: int = 10) -> List[Dict]:
    """
    title, artistDisplayName 에서 LIKE 검색.
    일치한 필드 수가 많은 순서로 정렬.
    """
    q = f"%{query}%"

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    rows = cur.execute(
        """
        SELECT objectID, title, artistDisplayName
        FROM artworks
        WHERE title LIKE ? OR artistDisplayName LIKE ?
        ORDER BY 
            (CASE WHEN title LIKE ? THEN 1 ELSE 0 END +
             CASE WHEN artistDisplayName LIKE ? THEN 1 ELSE 0 END) DESC
        LIMIT ?
        """,
        (q, q, q, q, limit)
    ).fetchall()

    conn.close()

    return [
        {
            "objectID": row[0],
            "title": row[1] or "",
            "artist": row[2] or "",
        }
        for row in rows
    ]
