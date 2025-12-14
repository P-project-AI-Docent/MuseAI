# backend/routers.py
from backend.tts import generate_tts, build_full_description_and_tts
from backend.stt import speech_to_text
from backend.wiki import wiki_summary_best_effort, _clean_artist_name
from fastapi import APIRouter, UploadFile, File, Query, Body, HTTPException
from PIL import Image
import io
import numpy as np
import ollama
import os
import re
# 로컬 Ollama 주소(수정 가능)
os.environ.setdefault("OLLAMA_HOST", "http://192.168.45.46:11434")

from backend.db import fetch_artwork
from backend.nlp_intent import predict_intent, predict_sub_intent_for_related
from backend.related_search import (
    related_by_text,      # SQL LIKE 기반 텍스트 검색
    related_by_image,     # CLIP(+LoRA) 기반 이미지 유사도 검색
    related_by_context    # BGE context 인덱스 기반 문맥 유사도
)
from backend.session_state import session_state
from backend.image_preprocess import process_and_search_yolo_enhanced  # YOLO+크롭+CLIP (업로드 전용)
from rag.rag_retrieval import search_chunks
from backend.wiki import wiki_summary_best_effort, _clean_artist_name

router = APIRouter(prefix="/api", tags=["ai-docent"])

# ------------------------------------------------------------
# 설명 스타일
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

# ------------------------------------------------------------
# 용어 정의 감지(“~가 무슨 뜻이야?”)
# ------------------------------------------------------------
TERM_TRIGGERS = ["무슨 뜻", "뜻이 뭐", "뜻이뭐", "뜻이 뭐야", "의미", "정의"]

def _is_term_definition(q: str) -> bool:
    q = q.strip()
    return any(t in q for t in TERM_TRIGGERS)

def _extract_term(q: str) -> str:
    # 1) 조사 제거 패턴    
    q = re.sub(r"(은|는|이|가|을|를|의|에|에서|으로|로)\b", "", q)

    # 2) “무슨 뜻이야” 패턴 제거
    q = re.sub(r"무슨 뜻.*", "", q)
    q = re.sub(r"뜻이 뭐.*", "", q)
    q = re.sub(r"뜻이뭐.*", "", q)
    q = re.sub(r"뜻.*", "", q)
    q = re.sub(r"의미.*", "", q)
    q = re.sub(r"정의.*", "", q)

    # 3) 특수문자 제거
    q = re.sub(r"[\"\'\?\!\.]", "", q)

    # 4) 양끝 공백 제거
    return q.strip()

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
# 2) 이미지 검색 (CLIP 기반) — 기본 검색은 CLIP(+LoRA)
#    업로드 사진은 /image/upload (YOLO 사용)로 처리
# ============================================================
@router.post("/image/search")
async def api_image_search(file: UploadFile = File(...), topk: int = 10):
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except:
        raise HTTPException(400, "Invalid image file")

    # 기본: CLIP(+LoRA) 유사도 검색
    results = related_by_image(img, topk=topk)

    # 필요시 YOLO로 강제 전환하려면 아래 라인으로 교체
    # results = process_and_search_yolo_enhanced(np.array(img), topk=topk)

    return {"results": results}

# ============================================================
# 3) 이미지 업로드 (YOLO 기반) — 배경 있는 실사 사진 정확도 향상
# ============================================================
@router.post("/image/upload")
async def api_image_upload(file: UploadFile = File(...), topk: int = 1):
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img_np = np.array(img)
    except:
        raise HTTPException(400, "Invalid image file")

    # 🔥 YOLO + 엣지 + 보정 + CLIP 검색 전체 파이프라인
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
    # 작품 메타 조회
    # ------------------------------
    meta = fetch_artwork(object_id)
    if not meta:
        raise HTTPException(404, "작품 정보를 찾을 수 없습니다.")

    title       = (meta.get("title") or "").strip()
    artist_raw  = (meta.get("artist") or "").strip()
    artist_clean= _clean_artist_name(artist_raw)
    date_txt    = (meta.get("date") or "").strip()
    medium      = (meta.get("medium") or "").strip()
    dept        = (meta.get("department") or "").strip()

    # ------------------------------
    # 의도 탐지
    # ------------------------------
    intent = predict_intent(question)
    session_state.update(session_id, intent=intent, last_question=question)

    # ---------------------------------------------------------
    # (A) 용어 정의
    # ---------------------------------------------------------
    if _is_term_definition(question):
        term = _extract_term(question)
        if term:
            wiki_txt, _ = wiki_summary_best_effort([term])
            if wiki_txt:
                return {"answer": f"‘{term}’의 뜻은 {wiki_txt}"}
            else:
                return {"answer": f"‘{term}’의 정의를 찾을 수 없습니다. 다른 표현으로도 물어봐 주세요."}

    # ============================================================
    # (0) 유사작품 기준 선택 대기 처리 (명시적 요청시에만)
    # ============================================================
    waiting = session_state.get(session_id, "waiting_similar_choice", False)
    if waiting:
        mode = predict_sub_intent_for_related(question)

        # --- 시각 기준(메트 DB 이미지 → CLIP-only) ---
        if mode == "visual":
            session_state.reset(session_id, "waiting_similar_choice")

            if not meta.get("localImagePath"):
                return {"answer": "이 작품의 로컬 이미지가 없어 시각적 유사 검색을 할 수 없습니다."}

            base_img = Image.open(meta["localImagePath"]).convert("RGB")
            # ✅ 업로드가 아닌 ‘작품 이미지’는 CLIP-only로!
            results = related_by_image(base_img, topk=3)

            return {
                "answer": "시각적으로 가장 유사한 작품 3개를 보여드릴게요.",
                "results": results
            }

        # --- 문맥 기준(BGE context) ---
        if mode == "context":
            session_state.reset(session_id, "waiting_similar_choice")
            context_results = related_by_context(question, topk=3)
            return {
                "answer": "내용·설명 측면에서 유사한 작품 3개를 알려드릴게요.",
                "results": context_results
            }

        return {"answer": "시각적인 기준인가요, 아니면 내용·문맥 기준인가요?"}

    # ============================================================
    # FALLBACK
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
    # 단순 즉답 Intent (+ 위키 보강)
    # ============================================================
    if intent == "artist_info":
        base = f"이 작품의 작가는 {artist_raw}입니다." if artist_raw else "작가 정보를 찾을 수 없습니다."
        wiki_txt, _ = wiki_summary_best_effort([artist_clean, artist_raw, title, medium, dept])
        if wiki_txt:
            return {"answer": f"{base} {wiki_txt}"}
        return {"answer": base}

    if intent == "date_query":
        base = f"이 작품은 {date_txt}에 제작되었습니다." if date_txt else "제작 시기를 찾을 수 없습니다."
        wiki_txt, _ = wiki_summary_best_effort([artist_clean, title, medium])
        if wiki_txt:
            return {"answer": f"{base} {wiki_txt}"}
        return {"answer": base}

    if intent == "medium_query":
        base = f"이 작품은 {medium} 재료를 사용해 제작되었습니다." if medium else "재료 정보를 찾을 수 없습니다."
        wiki_txt, _ = wiki_summary_best_effort([medium, title, artist_clean])
        if wiki_txt:
            return {"answer": f"{base} {wiki_txt}"}
        return {"answer": base}

    if intent == "metadata_query":
        base = (
            f"이 작품의 제목은 '{title}'이며, "
            f"{artist_raw}이(가) {date_txt}에 제작했습니다. "
            f"사용된 재료는 {medium}이며, "
            f"{dept} 부서에 소장되어 있습니다."
        )
        wiki_txt, _ = wiki_summary_best_effort([artist_clean, title, medium])
        if wiki_txt:
            return {"answer": f"{base} {wiki_txt}"}
        return {"answer": base}

    # ============================================================
    # artwork_overview / style_context
    # ============================================================
    if intent == "artwork_overview":
        # 작품 전반 요약: 메타 + 위키 + RAG
        wiki_txt, _ = wiki_summary_best_effort([artist_clean, title, medium, dept])
        rag_results = search_chunks(object_id, question, topk=4)
        rag_text = "\n".join(f"- {r['chunk']}" for r in rag_results) if rag_results else "설명 없음"

        prompt = f"""
        당신은 한국어만 사용하는 미술관 도슨트 AI입니다.
        외국어와 한자 사용 금지.

        [스타일]
        {style_prompt}

        [요청] 아래 정보를 모두 반영해 작품 개요를 5~7문장으로 자연스럽게 요약하세요.
        - 제목: {title}
        - 작가: {artist_raw}
        - 제작 시기: {date_txt}
        - 재료: {medium}
        - 부서: {dept}

        [위키 보강]
        {wiki_txt or "없음"}

        [RAG 참고]
        {rag_text}
        """
        resp = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
        return {"answer": resp["message"]["content"]}

    if intent == "style_context":
        # 사조/화풍/기법 설명
        wiki_txt, _ = wiki_summary_best_effort([title, artist_clean, medium, "사조", "화풍", "양식", "기법"])
        rag_results = search_chunks(object_id, question, topk=4)
        rag_text = "\n".join(f"- {r['chunk']}" for r in rag_results) if rag_results else "설명 없음"

        prompt = f"""
        당신은 한국어만 사용하는 미술관 도슨트 AI입니다.
        외국어/한자 금지.

        [스타일]
        {style_prompt}

        [요청]
        이 작품의 양식/사조/기법/표현 특성에 대해, 관람객이 이해하기 쉽게 5문장 내로 설명하세요.
        - 제목: {title}
        - 작가: {artist_raw}
        - 제작 시기: {date_txt}
        - 재료: {medium}

        [위키 보강]
        {wiki_txt or "없음"}

        [RAG 참고]
        {rag_text}
        """
        resp = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
        return {"answer": resp["message"]["content"]}

    # ============================================================
    # 관련 작품 추천 Intent
    #   - 명시적 요청시에만 버튼 표시
    #   - 시각적 유사: CLIP-only (메트 DB 이미지에 YOLO 금지)
    # ============================================================
    if intent in ["related_works", "similar_artwork"]:
        # 명시적 문구가 있을 때만 선택 버튼 제공
        if any(k in question for k in ["비슷한 작품", "유사한 작품", "비슷한 그림", "비슷한 거"]):
            session_state.set(session_id, "waiting_similar_choice", True)
            return {
                "answer": (
                    "어떤 기준으로 비슷한 작품을 찾을까요?\n"
                    "- 시각적인 유사도\n"
                    "- 내용/문맥 기반 유사도\n"
                    "원하시는 기준을 알려주세요!"
                )
            }
        return {"answer": "비슷한 작품을 원하시면 ‘비슷한 작품’이라고 말씀해 주세요. 기준(시각/문맥)도 함께 알려주시면 더 정확합니다."}

    # ============================================================
    # (마지막) 일반 설명: RAG + 위키 + 메타 통합
    # ============================================================
    wiki_txt, _ = wiki_summary_best_effort([artist_clean, title, medium])
    wiki_block = f"[위키 정보]\n{wiki_txt}\n" if wiki_txt else ""

    rag_results = search_chunks(object_id, question, topk=4)
    rag_text = "\n".join(f"- {r['chunk']}" for r in rag_results) if rag_results else "설명 없음"

    prompt = f"""
    당신은 한국어만 사용하는 미술관 도슨트 AI입니다.

    [스타일 규칙 — 최우선 적용]
    {style_prompt}

    [절대 금지 규칙 — 최우선]
    1) 영어 문장, 영어 단어, 영어 철자(A~Z / a~z) 절대 금지
    2) 외국어(프랑스어, 러시아어, 일본어, 중국어 등) 절대 금지
    3) 한자 절대 금지
    4) 같은 사실을 다른 표현으로 반복하는 행위 금지
    5) 동일하거나 유사한 의미를 문장만 바꿔 반복해서는 안 됨
    6) 사실을 알 수 없으면 추측하지 말고 "정보가 부족합니다"라고 말하기

    [설명 방식]
    - 한국어만 사용하여 자연스러운 서술형 문장으로 간결하게 설명합니다.
    - 작품의 핵심 요소(제목, 작가, 시대, 재료, 표현 방식, 특징)를 중심으로 설명합니다.
    - 어려운 용어는 쉬운 표현으로 풀어 설명합니다.
    - 문장은 짧고 명확하게 구성합니다.
    - 중복되는 내용은 절대 포함하지 않습니다.

    [사용자 질문]
    {question}

    [작품 정보 — 반드시 한국어로 변환하여 자연스럽게 설명할 것]
    제목: {meta['title']}
    작가: {meta['artist']}
    제작 시기: {meta['date']}
    재료: {meta['medium']}

    {wiki_block}

    [RAG 참고 설명]
    {rag_text}

    위 내용을 기반으로, 한국어만 사용하여 중복 없이 하나의 자연스러운 설명으로 정리하세요.
    """
    resp = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return {"answer": resp["message"]["content"]}

# ============================================================
# 6) 작품 metadata 반환
# ============================================================
@router.get("/artwork/{object_id}")
async def api_artwork_meta(object_id: int):
    meta = fetch_artwork(object_id)
    if not meta:
        raise HTTPException(404, "작품 정보를 찾을 수 없습니다.")
    return {
        "id": str(meta["objectID"]),
        "title": meta["title"],
        "artist": meta["artist"],
        "year": meta["date"],
        "description": meta["description"] or meta["desc_catalog"] or meta["desc_tech"] or "",
        "imageUrl": meta["localImagePath"] or meta["primaryImage"] or ""
    }
# ============================================================
# 7) TTS
# ============================================================

@router.post("/tts")
async def api_tts(payload: dict = Body(...)):
    text = (payload or {}).get("text", "").strip()
    if not text:
        raise HTTPException(400, "`text` is required")

    audio_url = generate_tts(text)
    return {"audioUrl": audio_url}


@router.get("/artwork/{object_id}/full-description")
async def api_full_desc(object_id: int):
    return build_full_description_and_tts(object_id)


# ============================================================
# 8) STT
# ============================================================

@router.post("/stt")
async def api_stt(file: UploadFile = File(...)):
    raw = await file.read()

    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(raw)

    text = speech_to_text(temp_path)

    return {"text": text}