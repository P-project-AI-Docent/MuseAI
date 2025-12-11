import os
import json
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import torch
import faiss
from PIL import Image

from transformers import CLIPModel, CLIPProcessor
from peft import PeftModel


# ============================================================
# PATH 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "clip_lora")                # LoRA 적용 모델 경로
BASE_CLIP_DIR = os.path.join(BASE_DIR, "clip_base")            # Base CLIP 경로

INDEX_DIR = os.path.join(BASE_DIR, "index_assets_lora")        # LoRA 인덱스 경로
FAISS_PATH = os.path.join(INDEX_DIR, "images.faiss")
PATHS_JSON = os.path.join(INDEX_DIR, "paths.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_proc = None
_index = None
_paths = None



# ============================================================
# 1) CLIP + LoRA 모델 로드
# ============================================================
def _load_model():
    global _model, _proc

    if _model is not None:
        return _model, _proc

    try:
        # base CLIP
        base_model = CLIPModel.from_pretrained(
            BASE_CLIP_DIR,
            local_files_only=True,
            ignore_mismatched_sizes=True
        ).to(DEVICE)

        # LoRA merge
        _model = PeftModel.from_pretrained(
            base_model,
            MODEL_DIR,
            is_trainable=False
        ).to(DEVICE).eval()

        # processor
        _proc = CLIPProcessor.from_pretrained(BASE_CLIP_DIR, local_files_only=True)
        print("[OK] CLIP + LoRA loaded")

    except Exception as e:
        print("[WARN] LoRA load failed:", e)
        print("[FALLBACK] base CLIP only")

        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
        _proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return _model, _proc



# ============================================================
# 2) FAISS index / paths.json 로드
# ============================================================
def _load_index():
    global _index, _paths

    if _index is not None:
        return _index, _paths

    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError(f"FAISS index not found: {FAISS_PATH}")

    if not os.path.exists(PATHS_JSON):
        raise FileNotFoundError(f"paths.json not found: {PATHS_JSON}")

    _index = faiss.read_index(FAISS_PATH)

    with open(PATHS_JSON, "r", encoding="utf-8") as f:
        _paths = json.load(f)

    print(f"[OK] Loaded FAISS index ({len(_paths)} items)")
    return _index, _paths



# ============================================================
# 3) 이미지 → CLIP 임베딩
# ============================================================
@torch.no_grad()
def _embed_image(img) -> np.ndarray:
    model, proc = _load_model()

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    if not isinstance(img, Image.Image):
        raise ValueError("Input must be PIL.Image.Image or numpy.ndarray")

    inputs = proc(images=img.convert("RGB"), return_tensors="pt").to(DEVICE)

    feats = model.get_image_features(**inputs)
    feats = torch.nn.functional.normalize(feats, p=2, dim=1)

    return feats.cpu().numpy().astype("float32")



# ============================================================
# 4) paths.json 항목 안전 파싱
# ============================================================
def _parse_path_entry(entry):
    """
    entry: str 또는 dict
    return: (path, objectID)
    """

    # Case 1: 문자열
    if isinstance(entry, str):
        path = entry
        try:
            oid = int(Path(entry).stem)
        except:
            oid = None
        return path, oid

    # Case 2: dict
    if isinstance(entry, dict):
        path = entry.get("path")
        oid = entry.get("objectID")

        # fallback: 파일명에서 추출
        if oid is None and isinstance(path, str):
            try:
                oid = int(Path(path).stem)
            except:
                oid = None

        return path, oid

    # 잘못된 형식
    return None, None



# ============================================================
# 5) 이미지 검색
# ============================================================
def search_image(img, topk: int = 10) -> List[Dict[str, Any]]:
    index, paths = _load_index()
    query_vec = _embed_image(img)

    D, I = index.search(query_vec, topk)

    results = []
    for score, idx in zip(D[0], I[0]):

        if idx >= len(paths):
            continue

        entry = paths[idx]
        img_path, oid = _parse_path_entry(entry)

        results.append({
            "score": float(score),
            "objectID": oid,
            "image_path": img_path,
            "raw_entry": entry
        })

    return results
