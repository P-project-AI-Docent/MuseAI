# backend/tts.py

import os
from gtts import gTTS
import ollama

from backend.db import fetch_artwork

# ------------------------------------------------------------
# 기본 설정 (오디오 파일 저장 위치)
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # backend/
AUDIO_DIR = os.path.join(BASE_DIR, "static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)


# ------------------------------------------------------------
# 1) 영어 → 한국어 번역 훅 (필요한 경우만 실행)
# ------------------------------------------------------------
def translate_to_korean(text: str) -> str:
    prompt = f"""
    아래 문장을 자연스럽고 정확한 한국어로 번역하세요.
    문장을 추가하거나 임의로 수정하지 마십시오.

    {text}
    """
    resp = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp["message"]["content"].strip()


# ------------------------------------------------------------
# 2) 작품 전체 설명 생성기
# ------------------------------------------------------------
def generate_full_description(object_id: int) -> str:
    meta = fetch_artwork(object_id)
    if not meta:
        return "작품 정보를 찾을 수 없습니다."

    # 기본 메타데이터
    title  = meta.get("title", "")
    artist = meta.get("artist", "")
    date   = meta.get("date", "")
    medium = meta.get("medium", "")
    desc_raw = (meta.get("met_description") or "").strip()

    # ● met_description이 영어면 번역 적용
    if desc_raw and any(ch.isascii() for ch in desc_raw):
        desc_kor = translate_to_korean(desc_raw)
    else:
        desc_kor = desc_raw or ""

    # ● 메타데이터를 자연스러운 문장으로 재구성
    basic_info = (
        f"이 작품의 제목은 '{title}'입니다. "
        f"작가는 {artist}입니다. "
        f"{date}에 제작되었고 "
        f"사용된 재료는 {medium}입니다."
    )


    # ● 전체 설명 구성 (도슨트 톤)
    prompt = f"""
    당신은 한국어만 사용하는 전문 도슨트 AI입니다.

    아래 정보를 바탕으로
    - 자연스럽고 매끄러운 한국어 문장
    - 같은 사실 반복 금지
    - 도슨트 말투
    - 영어/한자 금지

    하나의 완성된 설명을 작성하세요.

    [기본 정보]
    {basic_info}

    [작품 설명]
    {desc_kor}
    """

    resp = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return resp["message"]["content"].strip()


# ------------------------------------------------------------
# 3) TTS 생성기 — 최종 mp3 파일 생성
# ------------------------------------------------------------
def generate_tts(text: str, filename: str = "tts.mp3") -> str:
    abs_path = os.path.join(AUDIO_DIR, filename)

    # 1) 기본 gTTS 생성
    tts = gTTS(text=text, lang="ko")
    tts.save(abs_path)

    # 2) pydub을 이용해 50% 빠르게 변환
    from pydub import AudioSegment
    sound = AudioSegment.from_file(abs_path)
    faster_sound = sound.speedup(playback_speed=1.5)

    # 3) 기존 파일을 빠른 버전으로 덮어쓰기
    faster_sound.export(abs_path, format="mp3")

    # 프론트에서 사용할 URL 반환
    return f"/static/audio/{filename}"



# ------------------------------------------------------------
# 4) 전체 설명 + TTS 묶음 (API 라우터에서 간단하게 호출)
# ------------------------------------------------------------
def build_full_description_and_tts(object_id: int):
    text = generate_full_description(object_id)
    audio_url = generate_tts(text, filename=f"{object_id}.mp3")
    return {
        "text": text,
        "audioUrl": audio_url
    }
