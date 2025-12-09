"""
cv_clip/zero-shot_model.py

Zero-shot Image-to-Image Retrieval with Metadata
Force safetensors loading (fixes torch.load vulnerability)
"""

import os
os.environ["TRANSFORMERS_USE_SAFE_TENSORS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
from pathlib import Path
from typing import List
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import faiss
except ImportError:
    raise SystemExit("Please install faiss-cpu: pip install faiss-cpu")

from transformers import CLIPProcessor, CLIPModel


# ------------------------------------------------
# CONFIG
# ------------------------------------------------
IMAGE_DIR = os.environ.get("IMAGE_DIR", "./met20k/images")
INDEX_DIR = os.environ.get("INDEX_DIR", "./index_assets")
MODEL_NAME = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
METADATA_CSV = os.environ.get("METADATA_CSV", "./met20k/metadata_with_description.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOPK = 10


# ------------------------------------------------
# LOAD METADATA
# ------------------------------------------------
metadata_df = None
if Path(METADATA_CSV).exists():
    metadata_df = pd.read_csv(METADATA_CSV)
    if "objectID" in metadata_df.columns:
        metadata_df.set_index("objectID", inplace=True)


# ------------------------------------------------
# UTILITIES
# ------------------------------------------------
def list_images(folder: str) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [p for p in Path(folder).glob("*") if p.suffix.lower() in exts]


@torch.no_grad()
def embed_images(model, processor, paths: List[Path]) -> np.ndarray:
    embs = []
    for p in tqdm(paths, desc="Embedding images"):
        img = Image.open(p).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        feat = model.get_image_features(**inputs)
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        embs.append(feat.cpu().numpy())
    return np.vstack(embs).astype("float32")


def build_index(vectors: np.ndarray):
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vectors)
    return index


def save_index(index, meta_paths: List[str]):
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(Path(INDEX_DIR) / "images.faiss"))
    with open(Path(INDEX_DIR) / "paths.json", "w", encoding="utf-8") as f:
        json.dump(meta_paths, f, ensure_ascii=False, indent=2)


def load_index():
    index = faiss.read_index(str(Path(INDEX_DIR) / "images.faiss"))
    with open(Path(INDEX_DIR) / "paths.json", "r", encoding="utf-8") as f:
        paths = json.load(f)
    return index, paths


@torch.no_grad()
def embed_single_image(model, processor, img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    feat = model.get_image_features(**inputs)
    feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")


# ------------------------------------------------
# BUILD INDEX
# ------------------------------------------------
def cmd_build():
    paths = list_images(IMAGE_DIR)
    if not paths:
        raise SystemExit(f"No images found in {IMAGE_DIR}")

    print(f"Found {len(paths)} images. Building index...")

    model = CLIPModel.from_pretrained(
        MODEL_NAME,
        use_safetensors=True
    ).to(DEVICE)

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME,
        use_safetensors=True
    )

    vectors = embed_images(model, processor, paths)
    index = build_index(vectors)
    save_index(index, [str(p) for p in paths])

    print(f"Index saved to {INDEX_DIR}")


# ------------------------------------------------
# QUERY
# ------------------------------------------------
def cmd_query(img_path: str, topk: int = TOPK):
    index, paths = load_index()

    model = CLIPModel.from_pretrained(
        MODEL_NAME,
        use_safetensors=True
    ).to(DEVICE)

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME,
        use_safetensors=True
    )

    q = embed_single_image(model, processor, img_path)
    D, I = index.search(q, topk)

    print(f"\nTop-{topk} similar images to {img_path}:\n")

    for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
        image_path = paths[idx]
        oid = int(Path(image_path).stem) if Path(image_path).stem.isdigit() else None

        title = artist = date = medium = description = "N/A"

        # 메타데이터 매칭
        if metadata_df is not None and oid in metadata_df.index:
            row = metadata_df.loc[oid]
            title = str(row.get("title", "N/A"))
            artist = str(row.get("artistDisplayName", "N/A"))
            date = str(row.get("objectDate", "N/A"))
            medium = str(row.get("medium", "N/A"))

            desc_full = str(row.get("met_description", ""))
            description = (desc_full[:150] + "...") if len(desc_full) > 150 else desc_full

        print(f"{rank}. score={float(score):.4f}")
        print(f"   image: {image_path}")
        print(f"   title: {title}")
        print(f"   artist: {artist}")
        print(f"   date: {date}")
        print(f"   medium: {medium}")
        print(f"   description: {description}")
        print()


# ------------------------------------------------
# MAIN
# ------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--query", type=str)
    ap.add_argument("--topk", type=int, default=TOPK)
    args = ap.parse_args()

    if args.build:
        cmd_build()
    elif args.query:
        cmd_query(args.query, args.topk)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
