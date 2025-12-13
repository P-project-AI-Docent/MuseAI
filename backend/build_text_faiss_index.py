import os
import json
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import torch
import faiss
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from .db import get_artworks

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "clip_finetuned")
INDEX_DIR = os.path.join(BASE_DIR, "index_assets_finetuned")
FAISS_PATH = os.path.join(INDEX_DIR, "images.faiss")
PATHS_JSON = os.path.join(INDEX_DIR, "paths.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_proc = None
_index = None
_paths = None


# ----------------------------------------------------------
# CLIP 모델 + Processor
# ----------------------------------------------------------
def _load_model():
    global _model, _proc
    if _model is None:
        _model = CLIPModel.from_pretrained(MODEL_DIR, use_safetensors=True).to(DEVICE).eval()
        _proc = CLIPProcessor.from_pretrained(MODEL_DIR)
    return _model, _proc


# ----------------------------------------------------------
# FAISS 인덱스 로드
# ----------------------------------------------------------
def _load_index():
    global _index, _paths
    if _index is None:
        if not os.path.exists(FAISS_PATH) or not os.path.exists(PATHS_JSON):
            raise FileNotFoundError(f"Missing image FAISS index under {INDEX_DIR}")

        _index = faiss.read_index(FAISS_PATH)

        with open(PATHS_JSON, "r", encoding="utf-8") as f:
            _paths = json.load(f)

    return _index, _paths


# ----------------------------------------------------------
# 이미지 임베딩 추출
# ----------------------------------------------------------
@torch.no_grad()
def _embed_image(pil_img: Image.Image) -> np.ndarray:
    model, proc = _load_model()
    inputs = proc(images=pil_img.convert("RGB"), return_tensors="pt").to(DEVICE)
    feats = model.get_image_features(**inputs)
    feats = torch.nn.functional.normalize(feats, p=2, dim=1)
    return feats.cpu().numpy().astype("float32")


# ----------------------------------------------------------
# 이미지 검색
# ----------------------------------------------------------
def search_image(pil_img: Image.Image, topk: int = 10) -> List[Dict[str, Any]]:
    index, paths = _load_index()
    query_vec = _embed_image(pil_img)

    D, I = index.search(query_vec, topk)

    # objectID 추출 (파일 이름 기반)
    object_ids = []
    for idx in I[0]:
        stem = Path(paths[idx]).stem
        try:
            object_ids.append(int(stem))
        except:
            object_ids.append(None)

    # SQLite 메타데이터 조회
    meta_rows = get_artworks([oid for oid in object_ids if oid is not None])
    meta_by_id = {m["objectID"]: m for m in meta_rows}

    results = []
    for score, oid, idx in zip(D[0], object_ids, I[0]):
        results.append({
            "score": float(score),
            "objectID": oid,
            "image_path": paths[idx],
            "meta": meta_by_id.get(oid)
        })

    return results
import os
import json
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import torch
import faiss
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from .db import get_artworks

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "clip_finetuned")
INDEX_DIR = os.path.join(BASE_DIR, "index_assets_finetuned")
FAISS_PATH = os.path.join(INDEX_DIR, "images.faiss")
PATHS_JSON = os.path.join(INDEX_DIR, "paths.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_proc = None
_index = None
_paths = None


# ----------------------------------------------------------
# CLIP 모델 + Processor
# ----------------------------------------------------------
def _load_model():
    global _model, _proc
    if _model is None:
        _model = CLIPModel.from_pretrained(MODEL_DIR, use_safetensors=True).to(DEVICE).eval()
        _proc = CLIPProcessor.from_pretrained(MODEL_DIR)
    return _model, _proc


# ----------------------------------------------------------
# FAISS 인덱스 로드
# ----------------------------------------------------------
def _load_index():
    global _index, _paths
    if _index is None:
        if not os.path.exists(FAISS_PATH) or not os.path.exists(PATHS_JSON):
            raise FileNotFoundError(f"Missing image FAISS index under {INDEX_DIR}")

        _index = faiss.read_index(FAISS_PATH)

        with open(PATHS_JSON, "r", encoding="utf-8") as f:
            _paths = json.load(f)

    return _index, _paths


# ----------------------------------------------------------
# 이미지 임베딩 추출
# ----------------------------------------------------------
@torch.no_grad()
def _embed_image(pil_img: Image.Image) -> np.ndarray:
    model, proc = _load_model()
    inputs = proc(images=pil_img.convert("RGB"), return_tensors="pt").to(DEVICE)
    feats = model.get_image_features(**inputs)
    feats = torch.nn.functional.normalize(feats, p=2, dim=1)
    return feats.cpu().numpy().astype("float32")


# ----------------------------------------------------------
# 이미지 검색
# ----------------------------------------------------------
def search_image(pil_img: Image.Image, topk: int = 10) -> List[Dict[str, Any]]:
    index, paths = _load_index()
    query_vec = _embed_image(pil_img)

    D, I = index.search(query_vec, topk)

    # objectID 추출 (파일 이름 기반)
    object_ids = []
    for idx in I[0]:
        stem = Path(paths[idx]).stem
        try:
            object_ids.append(int(stem))
        except:
            object_ids.append(None)

    # SQLite 메타데이터 조회
    meta_rows = get_artworks([oid for oid in object_ids if oid is not None])
    meta_by_id = {m["objectID"]: m for m in meta_rows}

    results = []
    for score, oid, idx in zip(D[0], object_ids, I[0]):
        results.append({
            "score": float(score),
            "objectID": oid,
            "image_path": paths[idx],
            "meta": meta_by_id.get(oid)
        })

    return results
