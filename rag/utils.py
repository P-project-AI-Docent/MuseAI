import os

# ai_docent/ 기준 루트
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# -------------------------------------------
# 📌 BGE-M3 safetensors 모델 경로 (이미 존재하는 폴더)
# -------------------------------------------
EMBED_MODEL_DIR = os.path.join(BASE_DIR, "bge_safe")

# -------------------------------------------
# 📌 RAG 인덱스 저장 디렉토리 (현재 사용하는 위치)
# -------------------------------------------
ASSET_DIR = os.path.join(BASE_DIR, "rag_assets")

os.makedirs(ASSET_DIR, exist_ok=True)

# -------------------------------------------
# 📌 저장될 파일 경로 (실제 시스템이 읽는 위치와 통일)
# -------------------------------------------
FAISS_PATH = os.path.join(ASSET_DIR, "rag_index.faiss")
IDMAP_JSON = os.path.join(ASSET_DIR, "rag_idmap.json")
