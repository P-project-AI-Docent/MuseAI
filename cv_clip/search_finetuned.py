"""
cv_clip/search_finetuned.py — LoRA version (WORKING)
"""

import os
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import pandas as pd

from transformers import CLIPModel, CLIPProcessor
from peft import PeftModel


# ------------------------------------------------
# CONFIG
# ------------------------------------------------
MODEL_DIR = "./clip_lora"                   # LoRA adapter dir
BASE_CLIP_DIR = "/home/cvip-titan/sunwoo/ai_docent/clip_base"

INDEX_DIR = "./index_assets_lora"           # LoRA index folder
METADATA_CSV = "./met20k/metadata_with_description.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOPK = 10


# ------------------------------------------------
# Load Metadata
# ------------------------------------------------
metadata_df = pd.read_csv(METADATA_CSV)
metadata_df = metadata_df.set_index("objectID")


# ------------------------------------------------
# FAISS + PATHS
# ------------------------------------------------
def load_index():
    import faiss

    index = faiss.read_index(str(Path(INDEX_DIR) / "images.faiss"))

    with open(Path(INDEX_DIR) / "paths.json", "r", encoding="utf-8") as f:
        paths = json.load(f)

    return index, paths


# ------------------------------------------------
# Query Embedding
# ------------------------------------------------
@torch.no_grad()
def embed_query(model, processor, img_path):
    img = Image.open(img_path).convert("RGB")

    inputs = processor(
        images=img,
        return_tensors="pt"
    ).to(DEVICE)

    # Base CLIP + LoRA merged model
    feat = model.get_image_features(**inputs)
    feat = feat / feat.norm(p=2, dim=-1, keepdim=True)

    return feat.cpu().numpy().astype("float32")


# ------------------------------------------------
# Run Query
# ------------------------------------------------
def run_query(img_path, topk=TOPK):
    print("\nLoading LoRA fine-tuned CLIP...")

    # 1) Load base CLIP
    base_model = CLIPModel.from_pretrained(
        BASE_CLIP_DIR,
        local_files_only=True,
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    # 2) Load LoRA adapters
    model = PeftModel.from_pretrained(
        base_model,
        MODEL_DIR,
        is_trainable=False
    ).to(DEVICE).eval()

    # Processor comes from the base CLIP
    processor = CLIPProcessor.from_pretrained(
        BASE_CLIP_DIR,
        local_files_only=True
    )

    print("Loading FAISS index...")
    index, meta = load_index()

    print(f"\nRunning query on: {img_path}\n")
    q = embed_query(model, processor, img_path)
    D, I = index.search(q, topk)

    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
        entry = meta[idx]
        obj_id = entry["objectID"]
        image_path = entry["path"]

        # metadata fetch
        if obj_id in metadata_df.index:
            row = metadata_df.loc[obj_id]
        else:
            print(f"{rank}. [WARNING] objectID {obj_id} missing in metadata.")
            continue

        title = row.get("title", "")
        artist = row.get("artistDisplayName", "")
        date = row.get("objectDate", "")
        medium = row.get("medium", "")
        desc = str(row.get("met_description", ""))

        if len(desc) > 200:
            desc = desc[:200] + "..."

        print(f"{rank}. score={float(score):.4f}")
        print(f"   image: {image_path}")
        print(f"   objectID: {obj_id}")
        print(f"   title: {title}")
        print(f"   artist: {artist}")
        print(f"   date: {date}")
        print(f"   medium: {medium}")
        print(f"   description: {desc}\n")


# ------------------------------------------------
# MAIN
# ------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--topk", type=int, default=10)

    args = ap.parse_args()
    run_query(args.query, args.topk)
