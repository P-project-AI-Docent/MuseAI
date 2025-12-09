# rag/rag_build.py

import os
import json
import sqlite3
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

from .utils import EMBED_MODEL_DIR, FAISS_PATH, IDMAP_JSON
from backend.db import DB_PATH


# ====== 여기 중요! 반드시 변경 ======
CHUNK_SIZE = 80
CHUNK_OVERLAP = 20
# =====================================


# ----------------------------
# 1) 청크 생성 함수 (OVERLAP 적용)
# ----------------------------
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    step = size - overlap
    for i in range(0, len(words), step):
        yield " ".join(words[i:i + size])


# ----------------------------
# 2) 인덱스 생성
# ----------------------------
def build_rag_index():
    print("[1] Loading safe embedding model (bge-m3 safetensors only)…")
    model = SentenceTransformer(EMBED_MODEL_DIR, trust_remote_code=True)

    print("[2] Loading metadata database…")
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT objectID, met_description FROM artworks"
    ).fetchall()

    embeddings = []
    id_list = []

    print("[3] Creating chunks & embeddings…")

    for obj_id, desc in tqdm(rows):
        if not desc or not desc.strip():
            continue
        
        # ---- 수정된 chunking 적용 ----
        chunks = list(chunk_text(desc))

        for chunk in chunks:
            emb = model.encode(chunk, normalize_embeddings=True)
            embeddings.append(emb.astype("float32"))
            id_list.append({
                "objectID": obj_id,
                "chunk": chunk
            })

    if not embeddings:
        print("No embeddings created. Aborting.")
        return

    embeddings = np.vstack(embeddings)

    print("[4] Building FAISS index…")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print("[5] Saving FAISS index & ID map…")
    os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)

    faiss.write_index(index, FAISS_PATH)

    with open(IDMAP_JSON, "w", encoding="utf-8") as f:
        json.dump(id_list, f, ensure_ascii=False, indent=2)

    print("✔ RAG index build complete!")


if __name__ == "__main__":
    build_rag_index()
