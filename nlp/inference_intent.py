# nlp/inference_intent.py
# ---------------------------------------
# Intent Prediction (with Keyword Boosting)
# ---------------------------------------

import os
import json
import torch
import re
from transformers import AutoTokenizer
from nlp.train_intent_classifier import IntentClassifier, MODEL_NAME

# ---------------------------
# DEVICE 자동 선택
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE] Using {DEVICE}")

# ---------------------------
# 레이블 로드
# ---------------------------
LABEL_PATH = "nlp/intent_labels.json"
if not os.path.exists(LABEL_PATH):
    raise FileNotFoundError("intent_labels.json 을 찾을 수 없습니다.")

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)

id2label = {int(k): v for k, v in labels.items()}
label2id = {v: k for k, v in id2label.items()}


# ============================================================
# 🔍 Keyword Boosting Rules (최신 업데이트)
# ============================================================
KEYWORDS = {
    # 기본 설명
    "artwork_overview": {
        "words": ["소개", "개요", "요약", "전체적", "전반적", "설명"],
        "boost": 0.9
    },

    # 작가 정보
    "artist_info": {
        "words": ["작가", "예술가", "누구", "그린", "만든 사람", "화가"],
        "boost": 1.0
    },

    # 제작 시기
    "date_query": {
        "words": ["언제", "년도", "연도", "시기", "시대"],
        "boost": 0.9
    },

    # 재료
    "medium_query": {
        "words": ["재료", "소재", "재질", "무엇으로", "어떤 재료"],
        "boost": 0.9
    },

    # 메타데이터 정보
    "metadata_query": {
        "words": ["제목", "번호", "아이디", "크기", "정보"],
        "boost": 0.7
    },

    # ============================================================
    # 🔥 새로 추가되는 핵심 intent
    # ============================================================

    # 1단계: "유사한 작품 알려줘"
    "similar_artwork": {
        "words": ["비슷", "유사", "추천", "닮은", "같은 느낌", "비슷한 작품"],
        "boost": 1.2
    },

    # 2단계 선택지: 시각적
    "related_visual": {
        "words": ["시각", "비주얼", "색감", "비슷하게 생긴", "겉모습", "외형"],
        "boost": 1.3
    },

    # 2단계 선택지: 문맥/주제
    "related_context": {
        "words": ["주제", "내용", "맥락", "사조", "설명 기반", "문맥"],
        "boost": 1.2
    },

    # 스타일/기법
    "style_context": {
        "words": ["스타일", "양식", "화풍", "사조", "기법", "미술사"],
        "boost": 0.7
    },

    # fallback
    "fallback": {"words": [], "boost": 0.0},
}


# ============================================================
# 텍스트 정규화
# ============================================================
def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ============================================================
# Keyword Boosting
# ============================================================
def apply_keyword_boost(query: str, logits: torch.Tensor):
    query = normalize_text(query)
    tokens = query.split()

    logits = logits.clone()

    for intent, cfg in KEYWORDS.items():
        intent_id = label2id[intent]
        boost_val = cfg["boost"]

        for kw in cfg["words"]:
            if kw in query or kw in tokens:
                logits[0][intent_id] += boost_val

    return logits


# ============================================================
# 모델 로드
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    local_files_only=True
)

model = IntentClassifier(num_labels=len(id2label))
ckpt_path = "nlp/intent_classifier_best.pt"
if not os.path.exists(ckpt_path):
    ckpt_path = "nlp/intent_classifier.pt"

model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

MAX_LEN = 96


# ============================================================
# 예측 함수
# ============================================================
@torch.no_grad()
def predict_intent(text: str) -> str:
    text = normalize_text(text)

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )

    logits = model(
        enc["input_ids"].to(DEVICE),
        enc["attention_mask"].to(DEVICE)
    )

    # Keyword boosting 적용
    logits = apply_keyword_boost(text, logits)

    pred = logits.argmax(dim=1).item()
    return id2label[pred]


# ============================================================
# CLI 테스트
# ============================================================
if __name__ == "__main__":
    print("의도 분류 테스트 모드. exit 입력 시 종료.")
    while True:
        q = input("\n질문: ").strip()
        if q.lower() == "exit":
            break
        print(" → intent:", predict_intent(q))
