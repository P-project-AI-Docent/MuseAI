# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routers import router as api_router
import os


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Docent Backend API",
        description="NLP Intent + Text Search + Image Search + RAG API",
        version="1.0.0"
    )

    # -------------------------------
    # CORS 설정
    # -------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # -------------------------------
    # ⭐ TTS 오디오 파일 제공 폴더 mount
    # -------------------------------
    STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
    AUDIO_DIR = os.path.join(STATIC_DIR, "audio")

    # 폴더 없으면 생성
    os.makedirs(AUDIO_DIR, exist_ok=True)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # -------------------------------
    # ⭐ 정적 이미지 경로 mount
    # -------------------------------
    IMAGE_DIR = os.path.join(os.path.dirname(__file__), "met20k", "images")

    if not os.path.exists(IMAGE_DIR):
        raise RuntimeError(f"IMAGE_DIR not found: {IMAGE_DIR}")

    app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

    # -------------------------------
    # API 라우터
    # -------------------------------
    app.include_router(api_router)

    @app.get("/")
    async def root():
        return {"message": "AI Docent API is running"}

    return app


app = create_app()
