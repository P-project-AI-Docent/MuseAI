# tools/build_full_rag.py

import os
import sqlite3
import json
from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

# --------------------------------------------------------
# 경로 설정
# --------------------------------------------------------
ROOT_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(ROOT_DIR, "met20k", "metadata.db")

COL_CATALOG = "description_catalog"
COL_TECH = "description_technical"

RAG_DIR = os.path.join(ROOT_DIR, "rag_assets")
FAISS_PATH = os.path.join(RAG_DIR, "rag_index.faiss")
IDMAP_PATH = os.path.join(RAG_DIR, "rag_idmap.json")

EMBED_MODEL_DIR = "/home/cvip-titan/sunwoo/ai_docent/bge_safe"


# --------------------------------------------------------
# 청크 생성기
# --------------------------------------------------------
CHUNK_SIZE = 80
CHUNK_OVERLAP = 20

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    step = size - overlap
    for i in range(0, len(words), step):
        yield " ".join(words[i:i + size])


# --------------------------------------------------------
# TEXT CLEANER (footer 제거)
# --------------------------------------------------------
def clean_footer(text: str) -> str:
    CUT_KEYS = [
        "More Artwork",
        "More Artworks",
        "Related Content",
        "Related Artwork",
        "Related Essays",
    ]
    cleaned = text
    for key in CUT_KEYS:
        idx = cleaned.find(key)
        if idx != -1:
            cleaned = cleaned[:idx].trim()
    return cleaned.strip()


# --------------------------------------------------------
# PLAYWRIGHT 최신 스크래핑 로직
# --------------------------------------------------------
def click_tab(page, tab_name):
    selectors = [
        f'label:has-text("{tab_name}")',
        f'div.tabs_tabText__tixoU >> text="{tab_name}"',
        f'text="{tab_name}"'
    ]
    for sel in selectors:
        try:
            page.click(sel, timeout=2000)
            return True
        except:
            pass
    return False


def extract_tab_text(page):
    spans = page.query_selector_all('span[data-sentry-element="Markdown"]')
    if spans:
        return "\n".join([s.inner_text() for s in spans]).strip()
    try:
        panel = page.locator("div.tab-panel").first
        return panel.inner_text().strip()
    except:
        return ""


def scrape_one(objectID):
    """Catalogue Entry + Technical Notes 스크래핑"""
    url = f"https://www.metmuseum.org/art/collection/search/{objectID}"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            page.goto(url, timeout=70000, wait_until="domcontentloaded")
        except:
            browser.close()
            return "", ""

        # Catalogue Entry
        catalogue = ""
        if click_tab(page, "Catalogue Entry"):
            page.wait_for_timeout(1500)
            catalogue = clean_footer(extract_tab_text(page))

        # Technical Notes
        technical = ""
        if click_tab(page, "Technical Notes"):
            page.wait_for_timeout(1500)
            technical = clean_footer(extract_tab_text(page))

        browser.close()

        return catalogue, technical


# --------------------------------------------------------
# 2. SQLite 업데이트
# --------------------------------------------------------
def update_sqlite():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 컬럼 추가
    for col in (COL_CATALOG, COL_TECH):
        try:
            cur.execute(f"ALTER TABLE artworks ADD COLUMN {col} TEXT DEFAULT ''")
        except:
            pass

    conn.commit()

    rows = cur.execute("SELECT objectID FROM artworks").fetchall()

    print("\n[1] MetMuseum 페이지 스크래핑 시작…\n")

    for (objectID,) in tqdm(rows):
        catalogue, technical = scrape_one(objectID)

        cur.execute(f"""
            UPDATE artworks
            SET {COL_CATALOG}=?, {COL_TECH}=?
            WHERE objectID=?
        """, (catalogue, technical, objectID))

        conn.commit()

    conn.close()
    print("\n✔ SQLite 업데이트 완료!")


# --------------------------------------------------------
# 3. RAG 인덱스 생성
# --------------------------------------------------------
def build_rag_index():
    print("\n[2] RAG 인덱스 생성 시작…\n")

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(f"""
        SELECT objectID, met_description, {COL_CATALOG}, {COL_TECH}
        FROM artworks
    """).fetchall()
    conn.close()

    model = SentenceTransformer(EMBED_MODEL_DIR, trust_remote_code=True)

    embeddings = []
    id_list = []

    for obj_id, met_desc, catalog, tech in tqdm(rows):

        full_text = ""

        if met_desc and len(met_desc.strip()) > 0:
            full_text += met_desc + "\n\n"
        if catalog and len(catalog.strip()) > 0:
            full_text += catalog + "\n\n"
        if tech and len(tech.strip()) > 0:
            full_text += tech + "\n\n"

        if len(full_text.strip()) < 10:
            continue

        for chunk in chunk_text(full_text):
            emb = model.encode(chunk, normalize_embeddings=True)
            embeddings.append(emb.astype("float32"))
            id_list.append({
                "objectID": obj_id,
                "chunk": chunk
            })

    if not embeddings:
        print("❌ No embeddings created.")
        return

    embeddings = np.vstack(embeddings)
    dim = embeddings.shape[1]

    print("[3] FAISS 인덱스 저장…")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(RAG_DIR, exist_ok=True)

    faiss.write_index(index, FAISS_PATH)
    with open(IDMAP_PATH, "w", encoding="utf-8") as f:
        json.dump(id_list, f, ensure_ascii=False, indent=2)

    print("\n✔ RAG 빌드 완료!")
    print(f"FAISS → {FAISS_PATH}")
    print(f"IDMAP → {IDMAP_PATH}")


# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    print("\n====== FULL RAG PIPELINE START ======\n")

    update_sqlite()
    build_rag_index()

    print("\n====== ALL DONE! ======\n")


if __name__ == "__main__":
    main()
