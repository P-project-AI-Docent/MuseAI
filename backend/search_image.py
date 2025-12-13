# backend/search_image.py
import os
import json
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import torch
import faiss
from PIL import Image

from transformers import CLIPModel, CLIPProcessor
from peft import PeftModel, PeftConfig


# ============================================================
# PATH 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "clip_lora")
BASE_CLIP_DIR = os.path.join(BASE_DIR, "clip_base")

INDEX_DIR = os.path.join(BASE_DIR, "index_assets_lora")
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
        print("[INFO] Loading base CLIP…")
        base_model = CLIPModel.from_pretrained(
            BASE_CLIP_DIR,
            local_files_only=True
        ).to(DEVICE)

        print("[INFO] Loading LoRA adapter…")
        cfg = PeftConfig.from_pretrained(MODEL_DIR)

        _model = PeftModel.from_pretrained(
            base_model,
            MODEL_DIR
        ).to(DEVICE).eval()

        print("[OK] CLIP + LoRA loaded")

    except Exception as e:
        print("[WARN] LoRA load failed:", e)
        print("[FALLBACK] base CLIP only")

        _model = CLIPModel.from_pretrained(BASE_CLIP_DIR).to(DEVICE).eval()

    _proc = CLIPProcessor.from_pretrained(BASE_CLIP_DIR, local_files_only=True)
    return _model, _proc


# ============================================================
# 2) FAISS index 로드
# ============================================================
def _load_index():
    global _index, _paths

    if _index is not None:
        return _index, _paths

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

    inputs = proc(images=img.convert("RGB"), return_tensors="pt").to(DEVICE)

    feats = model.get_image_features(**inputs)
    feats = torch.nn.functional.normalize(feats, p=2, dim=1)

    return feats.cpu().numpy().astype("float32")


# ============================================================
# 4) paths.json entry 파싱
# ============================================================
def _parse_path_entry(entry):
    if isinstance(entry, str):
        path = entry
        try:
            oid = int(Path(entry).stem)
        except:
            oid = None
        return path, oid

    if isinstance(entry, dict):
        path = entry.get("path")
        oid = entry.get("objectID")

        if oid is None and isinstance(path, str):
            try:
                oid = int(Path(path).stem)
            except:
                oid = None

        return path, oid

    return None, None


# ============================================================
# 5) 이미지 검색 (순수 CLIP + FAISS)
# ============================================================
def search_image(img, topk: int = 10):
    index, paths = _load_index()

    # 🔥 auto_crop 절대 사용하지 않음 (전처리는 외부에서 처리)
    query_vec = _embed_image(img)

    D, I = index.search(query_vec, topk + 1)

    results = []
    for score, idx in zip(D[0], I[0]):

        if score > 0.9999:
            continue

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

    return results[:topk]
