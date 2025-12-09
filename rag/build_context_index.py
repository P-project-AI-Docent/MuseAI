# rag/build_context_index.py

import os
import json
import sqlite3
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))            # ai_docent/
DB_PATH = os.path.join(BASE_DIR, "met20k", "metadata.db")

ASSET_DIR = os.path.join(BASE_DIR, "rag_assets")
os.makedirs(ASSET_DIR, exist_ok=True)

INDEX_PATH = os.path.join(ASSET_DIR, "context_index.faiss")
IDMAP_PATH = os.path.join(ASSET_DIR, "context_idmap.json")

# BGE-M3 safetensors만 있는 안전 폴더
BGE_MODEL_DIR = os.path.join(BASE_DIR, "bge_safe")


# ---------------------------------------------------------
# 유틸
# ---------------------------------------------------------
def clean_text(s: str) -> str:
    if not s:
        return ""
    # 너무 긴 텍스트는 자른다. 문맥 대표성만 유지
    s = s.replace("\r", " ").replace("\n", " ").strip()
    return " ".join(s.split())[:4000]


def build_context_text(row: Tuple) -> str:
    """
    row 스키마:
      objectID, title, artistDisplayName, artistDisplayBio,
      objectDate, medium, department, primaryImage, localImagePath,
      met_description, description_catalog, description_technical
    """
    (_oid, _title, artist, artist_bio, obj_date,
     _medium, _dept, _pimg, _limg, met_desc, _cat, _tech) = row

    parts = []
    if artist:
        parts.append(f"작가 {artist}")
    if artist_bio:
        parts.append(f"작가 소개 {artist_bio}")
    if obj_date:
        parts.append(f"제작 시기 {obj_date}")
    if met_desc:
        parts.append(f"작품 설명 {met_desc}")

    text = " / ".join([clean_text(p) for p in parts if p])
    return text


def fetch_rows(limit: int = None) -> List[Tuple]:
    q = """
    SELECT
        objectID, title, artistDisplayName, artistDisplayBio,
        objectDate, medium, department, primaryImage, localImagePath,
        met_description, description_catalog, description_technical
    FROM artworks
    """
    if limit and limit > 0:
        q += f" LIMIT {int(limit)}"

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    rows = cur.execute(q).fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------
# 인덱스 생성
# ---------------------------------------------------------
def main(batch_size: int = 64, limit: int = None):
    print("=== 작품 단위 문맥 임베딩 INDEX 생성 시작 ===")
    rows = fetch_rows(limit=limit)
    print(f"총 작품 수: {len(rows)}")

    # 문맥 텍스트 준비
    objects: List[Dict] = []
    corpus: List[str] = []

    for r in rows:
        oid = r[0]
        text = build_context_text(r)
        if not text:
            # 문맥이 전혀 없으면 스킵
            continue
        objects.append({"objectID": int(oid)})
        corpus.append(text)

    if not corpus:
        raise RuntimeError("임베딩할 문맥 텍스트가 없습니다.")

    print(f"문맥 생성 완료: {len(corpus)}개")

    # 모델 로드
    print("BGE-M3 모델 로드 중...")
    model = SentenceTransformer(BGE_MODEL_DIR, trust_remote_code=True)

    # 임베딩
    print("임베딩 생성 중...")
    vecs: List[np.ndarray] = []
    for i in tqdm(range(0, len(corpus), batch_size)):
        batch = corpus[i:i + batch_size]
        emb = model.encode(batch, normalize_embeddings=True)
        vecs.append(emb.astype("float32"))

    embs = np.vstack(vecs)  # (N, D)
    n, d = embs.shape
    print(f"임베딩 완료: {embs.shape}")

    # FAISS IndexFlatIP (코사인 유사도와 동치, normalize 했기 때문)
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    faiss.write_index(index, INDEX_PATH)

    # idmap 저장
    with open(IDMAP_PATH, "w", encoding="utf-8") as f:
        json.dump(objects, f, ensure_ascii=False, indent=2)

    print(f"저장 완료\n- index: {INDEX_PATH}\n- idmap: {IDMAP_PATH}")
    print("=== 완료 ===")


if __name__ == "__main__":
    # 필요하면 여기서 limit 조절 가능
    # main(limit=5000)
    main()
