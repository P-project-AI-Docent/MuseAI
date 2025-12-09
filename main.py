# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# backend/routers.py 를 FastAPI에 등록
from backend.routers import router as api_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Docent Backend API",
        description="NLP Intent + Text Search + Image Search + RAG API",
        version="1.0.0"
    )

    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 필요하면 도메인 제한 가능
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API 라우터 등록
    app.include_router(api_router)

    @app.get("/")
    async def root():
        return {"message": "AI Docent API is running"}

    return app


app = create_app()
