# backend/routers.py

from fastapi import APIRouter, UploadFile, File, Query, Body, HTTPException
from PIL import Image
import io
import numpy as np
import ollama

from backend.db import fetch_artwork
from backend.nlp_intent import predict_intent, predict_sub_intent_for_related
from backend.related_search import (
    related_by_text,
    related_by_image,
    related_by_context   # ← 문맥 기반 유사 검색 추가
)
from backend.session_state import session_state
from backend.image_preprocess import process_and_search_yolo_enhanced
from rag.rag_retrieval import search_chunks


router = APIRouter(prefix="/api", tags=["ai-docent"])


# ------------------------------------------------------------
# 도슨트 설명 스타일
# ------------------------------------------------------------
STYLE_PROMPTS = {
    "kids": (
        "초등학생도 이해할 수 있도록 아주 쉬운 단어와 예시를 사용해서 설명하세요. "
        "어려운 전문용어는 모두 풀어 쓰고, 친절하고 재미있게 이야기하듯 설명하세요."
    ),
    "expert": (
        "미술사적 개념, 표현 기법, 시대적 배경, 사조, 미학적 분석을 포함해 "
        "전문가 수준의 깊이 있는 설명을 제공하세요. 논리적 구조를 유지하세요."
    ),
    "docent": (
        "전문 도슨트처럼 자연스럽고 친절하게, 이야기하듯 설명하세요. "
        "개념은 어렵지 않게 풀어주되, 흐름은 매끄럽고 생동감 있게 전달하세요."
    ),
}

DEFAULT_STYLE = "docent"


# ============================================================
# 1) Intent API
# ============================================================
@router.post("/intent")
async def api_intent(payload: dict):
    text = (payload or {}).get("text", "").strip()
    if not text:
        raise HTTPException(400, "`text` is required")
    return {"intent": predict_intent(text)}


# ============================================================
# 2) 이미지 검색 (CLIP + LoRA)
# ============================================================
@router.post("/image/search")
async def api_image_search(file: UploadFile = File(...), topk: int = 10):
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except:
        raise HTTPException(400, "Invalid image file")

    results = related_by_image(img, topk=topk)
    return {"results": results}


# ============================================================
# 3) YOLO 기반 이미지 업로드 검색
# ============================================================
@router.post("/image/upload")
async def api_image_upload(file: UploadFile = File(...), topk: int = 1):
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_np = np.array(img)
    except:
        raise HTTPException(400, "Invalid image file")

    results = process_and_search_yolo_enhanced(img_np, topk=topk)
    return {"results": results}


# ============================================================
# 4) 텍스트 기반 검색 (SQL LIKE)
# ============================================================
@router.get("/text/search")
async def api_text_search(q: str = Query(...), limit: int = 50):
    results = related_by_text(q, topk=limit)
    return {"results": results}


# ============================================================
# 5) Chat API
# ============================================================
@router.post("/chat")
async def api_chat(payload: dict = Body(...)):

    # ------------------------------
    # 입력 파싱
    # ------------------------------
    question   = (payload or {}).get("question", "").strip()
    object_id  = (payload or {}).get("objectID", None)
    style      = (payload or {}).get("style", DEFAULT_STYLE)
    session_id = (payload or {}).get("sessionId", "default")

    if not question:
        raise HTTPException(400, "`question` is required")
    if object_id is None:
        raise HTTPException(400, "`objectID` is required")

    style_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS[DEFAULT_STYLE])

    # ------------------------------
    # 작품 Metadata 조회
    # ------------------------------
    meta = fetch_artwork(object_id)
    if not meta:
        raise HTTPException(404, "작품 정보를 찾을 수 없습니다.")

    # ------------------------------
    # Intent 분석
    # ------------------------------
    intent = predict_intent(question)
    session_state.update(session_id, intent=intent, last_question=question)

    # ============================================================
    # (0) 유사작품 기준 선택 대기 상태 처리
    # ============================================================
    waiting = session_state.get(session_id, "waiting_similar_choice", False)

    if waiting:
        mode = predict_sub_intent_for_related(question)

        # --- 시각적 기준 선택 ---
        if mode == "visual":
            session_state.reset(session_id, "waiting_similar_choice")

            base_img = Image.open(meta["localImagePath"])
            results = related_by_image(base_img, topk=3)

            return {
                "answer": "시각적으로 가장 유사한 작품 3개를 보여드릴게요.",
                "results": results
            }

        # --- 문맥 기준 선택 ---
        if mode == "context":
            session_state.reset(session_id, "waiting_similar_choice")

            context_results = related_by_context(question, topk=3)
            return {
                "answer": "내용·설명 측면에서 유사한 작품 3개를 알려드릴게요.",
                "results": context_results
            }

        return {
            "answer": "시각적인 기준인가요, 아니면 내용·문맥 기준인가요?"
        }

    # ============================================================
    # FALLBACK 처리
    # ============================================================
    if intent == "fallback":
        cnt = session_state.increment(session_id, "fallback_count")

        if cnt == 1:
            return {"answer": "조금만 더 구체적으로 질문해주실 수 있을까요?"}

        if cnt == 2:
            return {
                "answer": (
                    "이 작품에서 무엇이 궁금하신가요?\n"
                    "예를 들어:\n"
                    "- 언제 만들어졌나요?\n"
                    "- 재료가 무엇인가요?\n"
                    "- 작가는 누구인가요?\n"
                    "처럼 말씀해주시면 정확히 도와드릴 수 있어요!"
                )
            }

        session_state.reset(session_id, "fallback_count")
        return {
            "answer": "이해를 돕기 위해 기본적인 작품 정보를 먼저 안내해드릴게요.",
            "url": f"https://www.metmuseum.org/art/collection/search/{object_id}"
        }

    session_state.reset(session_id, "fallback_count")

    # ============================================================
    # 단순 즉답 Intent 처리
    # ============================================================
    if intent == "artist_info":
        return {"answer": f"이 작품의 작가는 {meta['artist']}입니다."}

    if intent == "date_query":
        return {"answer": f"이 작품은 {meta['date']}에 제작되었습니다."}

    if intent == "medium_query":
        return {"answer": f"이 작품은 {meta['medium']} 재료를 사용해 제작되었습니다."}

    if intent == "metadata_query":
        return {
            "answer": (
                f"이 작품의 제목은 '{meta['title']}'이며, "
                f"{meta['artist']}이(가) {meta['date']}에 제작했습니다. "
                f"사용된 재료는 {meta['medium']}이며, "
                f"{meta['department']} 부서에 소장되어 있습니다."
            )
        }

    # ============================================================
    # 관련 작품 추천 Intent
    # ============================================================
    if intent in ["related_works", "similar_artwork"]:
        session_state.set(session_id, "waiting_similar_choice", True)

        return {
            "answer": (
                "어떤 기준으로 비슷한 작품을 찾을까요?\n"
                "- 시각적인 유사도\n"
                "- 내용/설명 기반 유사도\n"
                "원하시는 기준을 알려주세요!"
            )
        }

    # ============================================================
    # RAG + LLM 기반 설명 생성
    # ============================================================
    rag_results = search_chunks(object_id, question, topk=4)
    rag_text = "\n".join(f"- {r['chunk']}" for r in rag_results) if rag_results else "설명 없음"

    # LLM Prompt
    prompt = f"""
    당신은 한국어만 사용하는 미술관 도슨트 AI입니다.
    절대 영어로 답변하지 마세요.  
    <context> 내부 문서가 영어로 되어 있어도, 반드시 자연스러운 한국어로 재해석하여 설명해야 합니다.  
    영어 문장을 그대로 사용하거나 섞지 마세요.
    다른 언어(러시아어, 중국어, 일본어 등)는 절대 사용하지 마세요.
    작품 설명 중 외국어가 있다면 한국어로 자연스럽게 바꿔서 설명하세요.
    한자도 사용하지 마세요.

    설명 스타일:
    {style_prompt}

    사용자 질문: {question}

    <context>
    이 작품의 제목은 "{meta['title']}"이며,
    작가는 {meta['artist']}입니다.
    제작 시기는 {meta['date']}이고,
    사용된 재료는 {meta['medium']}입니다.

    RAG 설명:
    {rag_text}
    </context>

    위 정보를 모두 반영해,
    자연스럽고 매끄럽고 줄글 형태로 설명하세요.
    """

    resp = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = resp["message"]["content"]
    return {"answer": answer}

# [ADD] 작품 메타데이터 조회
@router.get("/artwork/{object_id}")
async def api_artwork_meta(object_id: int):
    meta = fetch_artwork(object_id)
    if not meta:
        raise HTTPException(404, "작품 정보를 찾을 수 없습니다.")
    # 프론트 타입과 맞춰서 필드 구성
    return {
        "id": str(meta["objectID"]),
        "title": meta["title"],
        "artist": meta["artist"],
        "year": meta["date"],
        "description": meta["description"] or meta["desc_catalog"] or meta["desc_tech"] or "",
        # 이미지 우선순위: localImagePath → primaryImage
        "imageUrl": meta["localImagePath"] or meta["primaryImage"] or ""
    }
