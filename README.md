# 📘 AI Docent — Real-Time Museum Guide System  
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)]()  
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)]()  
[![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react)]()  
[![License](https://img.shields.io/badge/License-CC--BY--NC--4.0-yellow)]()  
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

📱 **실시간 작품 인식 기반 AI 도슨트 시스템**  
🎨 YOLO + CLIP 기반 작품 인식  
🧠 RAG + Llama3 AI 설명  
🎤 STT(음성 입력) · 🔊 TTS(음성 안내)  
📚 Wikipedia 연동 자동 설명 강화  

---

## 📑 Table of Contents
- [Overview](#overview)  
- [Features](#features)  
- [System Architecture](#system-architecture)  
- [Tech Stack](#tech-stack)  
- [Directory Structure](#directory-structure)  
- [Installation](#installation)  
- [Model Setup](#model-setup)  
- [Environment Variables](#environment-variables)  
- [Running the Project](#running-the-project)  
- [API Documentation](#api-documentation)  
- [Frontend UX Flow](#frontend-ux-flow)  
- [Performance Tips](#performance-tips)  
- [Future Improvements](#future-improvements)  
- [License](#license)

---

## 🧭 Overview
**AI Docent**는 카메라로 작품을 비추면 작품을 실시간 인식하고,  
사용자의 질의에 따라 RAG + LLM 기반 설명을 제공하는  
**모바일 우선 AI 도슨트 시스템**입니다.

- YOLO + CLIP 기반 작품 인식  
- Wikipedia 및 RAG 기반 설명 강화  
- TTS(음성 안내)  
- STT(음성 입력)  
- 실시간 카메라 모드 지원  

---

## 🚀 Features
- 📷 **실시간 작품 인식** (YOLO → CLIP)
- 💬 **LLM 기반 QnA** (한국어 전용)
- 📚 **Wikipedia 요약 자동 연결**
- 🔍 **듀얼 유사 작품 추천**  
  - 시각 기반 (CLIP)  
  - 문맥 기반 (BGE Embedding)
- 🔊 **TTS 전체 설명 생성**
- 🎤 **STT 음성 입력 (Vosk)**
- 🧠 **RAG 기반 설명 강화**
- 📱 **모바일 UI 친화적 React 프론트엔드**

---

## 🏗 System Architecture
```
Frontend (React)
 ├─ Camera Live Preview
 ├─ STT Button
 ├─ Chat UI
 └─ TTS Player
        │
        ▼
Backend (FastAPI)
 ├─ YOLO Preprocess
 ├─ CLIP Image Retrieval
 ├─ RAG (FAISS + SQLite)
 ├─ Llama3 via Ollama
 ├─ gTTS Audio Builder
 └─ Vosk STT Engine
        │
        ▼
Local Assets
 ├─ Models
 ├─ Images
 └─ Indexes
```

---

## 🧰 Tech Stack

### **Frontend**
- React + Vite + TypeScript  
- TailwindCSS  
- WebRTC Camera  
- Web Speech API(TTS)  
- Custom Audio Player  

### **Backend**
- FastAPI  
- YOLOv8  
- CLIP (OpenAI)  
- SentenceTransformers + FAISS  
- Llama3 (Ollama)  
- gTTS  
- Vosk STT  
- SQLite DB  

---

## 📂 Directory Structure
```
ai_docent/
│
├── backend/
│   ├── routers.py
│   ├── db.py
│   ├── stt.py
│   ├── tts.py
│   ├── related_search.py
│   └── session_state.py
│
├── rag/
│   └── rag_retrieval.py
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── index.html
│
├── clip_base/
├── clip_lora/
├── bge_safe/
├── met20k/
├── index_assets/
└── main.py
```

---

## ⚙️ Installation

### Backend
```bash
conda create -n aidocent python=3.10
conda activate aidocent
pip install -r requirements.txt
```

### Frontend
```bash
cd frontend
npm install
```

---

## 📥 Model Setup (Required Downloads)
다음 모델들은 용량이 크기 때문에 GitHub 저장소에는 포함되지 않습니다.  
직접 다운로드해서 아래 경로에 배치해야 합니다.

| Model | Directory | Required |
|-------|-----------|----------|
| CLIP Base | `/clip_base` | ✔ |
| CLIP LoRA | `/clip_lora` | ✔ |
| BGE Embedding | `/bge_safe` | ✔ |
| MET20K 이미지 | `/met20k/images` | ✔ |
| FAISS Index | `/index_assets` | ✔ |
| Vosk STT | `/stt/model/` | ✔ |

### Example: Vosk STT download
```bash
wget https://alphacephei.com/vosk/models/vosk-model-small-ko-0.22.zip
unzip vosk-model-small-ko-0.22.zip
mv vosk-model-small-ko-0.22 stt/model/
```

---

## 🔐 Environment Variables

### Backend
```
export OLLAMA_HOST=http://YOUR_SERVER_IP:11434
```

### Frontend
`src/components/...` 내부에서 다음을 수정:
```
const API_BASE = "https://YOUR-SERVER-IP:8001";
```

---

## ▶️ Running the Project

### Backend (HTTPS)
```bash
uvicorn main:app --host 0.0.0.0 --port 8001 \
  --ssl-keyfile ./localhost+2-key.pem \
  --ssl-certfile ./localhost+2.pem
```

### Frontend
```bash
npm run dev
```

---

## 📡 API Documentation

| Endpoint | Description |
|---------|-------------|
| **POST /api/image/upload** | YOLO + CLIP 기반 작품 인식 |
| **POST /api/chat** | LLM 기반 질의응답 & 위키 & RAG |
| **GET /api/artwork/{id}** | 작품 메타데이터 |
| **GET /api/artwork/{id}/full-description** | 전체 TTS 설명 생성 |
| **POST /api/stt** | 음성 → 텍스트 |
| **POST /api/tts** | 텍스트 → 음성 |

---

## 📱 Frontend UX Flow
1. **카메라 실행 → 작품 인식**  
2. 작품 정보 카드 출력  
3. “전체 설명 듣기(TTS)”  
4. QnA 모드 진입  
5. STT 버튼으로 음성 질문  
6. 유사 작품 추천(시각/문맥 기반)  

---

## 📄 License
This project is released under **CC-BY-NC 4.0**.

---

