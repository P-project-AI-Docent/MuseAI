import os
import json
import faiss
import numpy as np

from backend.search_text import search_text
from backend.search_image import search_image
from sentence_transformers import SentenceTransformer


# ============================================================
# 1) 기존 LIKE 검색
# ============================================================
def related_by_text(query: str, topk: int = 3):
    return search_text(query, limit=topk)


# ============================================================
# 2) 이미지 기반 검색
# ============================================================
def related_by_image(image, topk: int = 3):
    return search_image(image, topk=topk)


# ============================================================
# 3) 문맥 기반 검색 (context_index)
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "rag_assets")

CONTEXT_INDEX_PATH = os.path.join(ASSET_DIR, "context_index.faiss")
CONTEXT_IDMAP_PATH = os.path.join(ASSET_DIR, "context_idmap.json")

BGE_MODEL_DIR = os.path.join(BASE_DIR, "bge_safe")

# Lazy-loaded objects
_context_index = None
_context_idmap = None
_bge_model = None


def _load_context_index():
    """context_index.faiss / idmap.json / bge 모델 로드"""
    global _context_index, _context_idmap, _bge_model

    # FAISS index
    if _context_index is None:
        _context_index = faiss.read_index(CONTEXT_INDEX_PATH)

    # idmap
    if _context_idmap is None:
        with open(CONTEXT_IDMAP_PATH, "r", encoding="utf-8") as f:
            _context_idmap = json.load(f)

    # BGE model
    if _bge_model is None:
        _bge_model = SentenceTransformer(
            BGE_MODEL_DIR,
            trust_remote_code=True
        )


def _get_idmap_item(faiss_idx: int):
    """
    idmap이 list 형태든 dict 형태든 모두 지원
    """
    # list 형태: [{objectID: ...}, ...]
    if isinstance(_context_idmap, list):
        return _context_idmap[faiss_idx]

    # dict 형태: {"0": {...}, "1": {...}}
    key = str(faiss_idx)
    if key in _context_idmap:
        return _context_idmap[key]

    raise KeyError(f"Invalid faiss index {faiss_idx} in context_idmap.")


def related_by_context(query: str, topk: int = 3):
    """
    문맥 기반 유사도 검색
    - BGE-M3로 문맥 임베딩한 전체 작품 설명에서 유사도 검색
    """
    _load_context_index()

    # Query embedding
    qvec = _bge_model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    # FAISS similarity search
    sims, idxs = _context_index.search(qvec, topk)

    results = []
    for score, faiss_idx in zip(sims[0], idxs[0]):
        item = _get_idmap_item(int(faiss_idx))
        results.append({
            "score": float(score),
            "objectID": item["objectID"],
            "source": "context"
        })

    return results
