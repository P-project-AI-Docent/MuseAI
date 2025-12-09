import json
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer


# ============================================================
# 1) 경로 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # ai_docent/
ASSET_DIR = os.path.join(BASE_DIR, "rag_assets")

INDEX_PATH = os.path.join(ASSET_DIR, "rag_index.faiss")
IDMAP_PATH = os.path.join(ASSET_DIR, "rag_idmap.json")

BGE_MODEL_DIR = os.path.join(BASE_DIR, "bge_safe")


# ============================================================
# 2) Lazy Load
# ============================================================
FAISS_INDEX = None
IDMAP = None
BGE_MODEL = None


def _lazy_load():
    """FAISS index + idmap + BGE 모델을 단 1회만 로드"""
    global FAISS_INDEX, IDMAP, BGE_MODEL

    # Load idmap.json
    if IDMAP is None:
        with open(IDMAP_PATH, "r", encoding="utf-8") as f:
            IDMAP = json.load(f)

    # Load FAISS
    if FAISS_INDEX is None:
        FAISS_INDEX = faiss.read_index(INDEX_PATH)

    # Load BGE-M3 model
    if BGE_MODEL is None:
        BGE_MODEL = SentenceTransformer(BGE_MODEL_DIR, trust_remote_code=True)


# ============================================================
# 3) 텍스트 → BGE-M3 임베딩
# ============================================================
def embed_rag_text(text: str):
    """텍스트를 BGE-M3 임베딩으로 변환"""
    _lazy_load()
    vec = BGE_MODEL.encode([text], normalize_embeddings=True)
    return vec.astype("float32")  # shape = (1, 1024)


# ============================================================
# 4) 안전하게 chunk_info 가져오는 함수
# ============================================================
def _safe_get_chunk_info(faiss_idx):
    """
    idmap이 list 또는 dict 둘 다 지원.
    list: [ {...}, {...}, ... ]
    dict: { "0": {...}, "1": {...} }
    """

    # list 타입
    if isinstance(IDMAP, list):
        if 0 <= faiss_idx < len(IDMAP):
            return IDMAP[faiss_idx]
        return None

    # dict 타입
    if isinstance(IDMAP, dict):
        key = str(faiss_idx)
        return IDMAP.get(key)

    return None


# ============================================================
# 5) RAG 검색 (텍스트 입력)
# ============================================================
def search_chunks(object_id: int, query: str, topk: int = 4):
    """
    object_id: 설명할 작품 ID
    query: 사용자 질문
    """

    _lazy_load()

    # 1) BGE-M3 임베딩
    query_vec = embed_rag_text(query)

    # 2) FAISS 검색
    sims, idxs = FAISS_INDEX.search(query_vec, topk)

    # 3) objectID 기반 필터링
    results = []

    for faiss_idx, score in zip(idxs[0], sims[0]):

        chunk_info = _safe_get_chunk_info(int(faiss_idx))
        if not chunk_info:
            continue

        # chunk_info["objectID"] 정규화 (str/int 혼합 대비)
        cid = str(chunk_info.get("objectID"))
        oid = str(object_id)

        if cid == oid:
            results.append({
                "chunk": chunk_info.get("text", ""),
                "score": float(score)
            })

    return results
