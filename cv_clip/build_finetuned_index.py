"""
cv_clip/build_finetuned_index.py — LoRA version (WORKING)
"""

import os
import json
from pathlib import Path
from typing import List

import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

try:
    import faiss
except ImportError:
    raise SystemExit("pip install faiss-cpu 필요!")

from transformers import CLIPModel, CLIPProcessor
from peft import PeftModel


# ------------------------------------------------
# CONFIG
# ------------------------------------------------
IMAGE_DIR = "./met20k/images"

# LoRA 결과 저장 폴더
MODEL_DIR = "./clip_lora"

# base CLIP 저장 위치 (Titan)
BASE_CLIP_DIR = "/home/cvip-titan/sunwoo/ai_docent/clip_base"

# 새로운 인덱스 저장 위치
INDEX_DIR = "./index_assets_lora"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
IMG_SIZE = 224


# ------------------------------------------------
# UTILITIES
# ------------------------------------------------
def list_images(folder: str) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted([p for p in Path(folder).glob("*") if p.suffix.lower() in exts])


def load_image(path: Path):
    try:
        img = Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")
    return img


@torch.no_grad()
def embed_images(model, processor, paths: List[Path]) -> np.ndarray:
    vectors = []

    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="Embedding images"):
        batch_paths = paths[i : i + BATCH_SIZE]
        imgs = [load_image(p) for p in batch_paths]

        inputs = processor(
            images=imgs,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)

        # ⭐ CLIP Vision Encoder + LoRA Adapter
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)

        vectors.append(feats.cpu().numpy())

    return np.concatenate(vectors, axis=0).astype("float32")


def build_index(vectors: np.ndarray):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_index(index, paths: List[Path]):
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, f"{INDEX_DIR}/images.faiss")

    meta = [{"path": str(p), "objectID": int(p.stem)} for p in paths]
    with open(f"{INDEX_DIR}/paths.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n🔥 Saved FAISS index -> {INDEX_DIR}/images.faiss")
    print(f"🔥 Saved metadata -> {INDEX_DIR}/paths.json")


# ------------------------------------------------
# MAIN
# ------------------------------------------------
def main():
    print("Loading CLIP base + LoRA adapters...")

    # 1) Base CLIP 로드
    base_model = CLIPModel.from_pretrained(
        BASE_CLIP_DIR,
        local_files_only=True,
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    # 2) LoRA Adapter merge
    model = PeftModel.from_pretrained(
        base_model,
        MODEL_DIR,
        is_trainable=False
    ).to(DEVICE).eval()

    # processor는 Base CLIP 기반
    processor = CLIPProcessor.from_pretrained(
        BASE_CLIP_DIR,
        local_files_only=True
    )

    paths = list_images(IMAGE_DIR)
    print(f"\nFound {len(paths)} images. Embedding...\n")

    vectors = embed_images(model, processor, paths)

    index = build_index(vectors)

    save_index(index, paths)


if __name__ == "__main__":
    main()
