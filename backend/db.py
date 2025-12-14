import os
import sqlite3

# 실제 메타데이터 DB 위치 = met20k/metadata.db
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "met20k", "metadata.db")

# DB 존재 여부 체크 (중요)
if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"[ERROR] metadata.db not found at: {DB_PATH}")


def fetch_artwork(object_id: int):
    """objectID를 기반으로 artworks 테이블에서 메타데이터 조회"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.execute("""
            SELECT objectID, title, artistDisplayName, artistDisplayBio,
                   objectDate, medium, department,
                   primaryImage, localImagePath,
                   met_description, description_catalog, description_technical
            FROM artworks
            WHERE objectID = ?
        """, (object_id,))

        row = cur.fetchone()

    finally:
        conn.close()

    if not row:
        return None

    return {
        "objectID": row[0],
        "title": row[1] or "",
        "artist": row[2] or "",
        "bio": row[3] or "",
        "date": row[4] or "",
        "medium": row[5] or "",
        "department": row[6] or "",
        "primaryImage": row[7] or "",
        "localImagePath": row[8] or "",
        "description": row[9] or "",
        "desc_catalog": row[10] or "",
        "desc_tech": row[11] or "",
    }
