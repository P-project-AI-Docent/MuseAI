# backend/nlp_intent.py

import os
import json
import re
import torch
from transformers import AutoTokenizer

# 학습한 모델
from nlp.train_intent_classifier import IntentClassifier, MODEL_NAME


# ==========================================
# 경로 설정
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # ai_docent/
NLP_DIR = os.path.join(BASE_DIR, "nlp")
LABEL_PATH = os.path.join(NLP_DIR, "intent_labels.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# 라벨 로딩
# ==========================================
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)

id2label = {int(k): v for k, v in labels.items()}          # {0:"artist_info", 1:"date_query", ...}
label2id = {v: k for k, v in id2label.items()}


# ==========================================
# Keyword Boosting (의도 라벨 기반)
# ==========================================
KEYWORDS = {
    "artwork_overview": {
        "words": ["소개", "개요", "요약", "전체적", "전반적", "설명"],
        "boost": 0.9
    },
    "artist_info": {
        "words": ["작가", "예술가", "누구", "그린", "만든 사람", "화가"],
        "boost": 1.0
    },
    "date_query": {
        "words": ["언제", "년도", "연도", "시기", "시대"],
        "boost": 0.9
    },
    "medium_query": {
        "words": ["재료", "소재", "재질", "무엇으로", "어떤 재료"],
        "boost": 0.9
    },
    "metadata_query": {
        "words": ["제목", "번호", "아이디", "크기", "정보"],
        "boost": 0.7
    },
    "related_works": {
        "words": ["비슷", "유사", "추천", "같은 느낌"],
        "boost": 0.6
    },
    "style_context": {
        "words": ["스타일", "양식", "화풍", "사조", "기법", "미술사"],
        "boost": 0.7
    },
    # fallback
    "fallback": {"words": [], "boost": 0.0},
}


# ==========================================
# Tokenizer & Intent Model
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    local_files_only=True
)

_model = None


def _load_model():
    """Intent Classification 모델 로드"""
    global _model
    if _model is None:
        ckpt = os.path.join(NLP_DIR, "intent_classifier_best.pt")
        if not os.path.exists(ckpt):
            ckpt = os.path.join(NLP_DIR, "intent_classifier.pt")

        model = IntentClassifier(num_labels=len(id2label))
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        _model = model

    return _model


# ==========================================
# 텍스트 정규화
# ==========================================
def normalize(text: str):
    text = text.strip()
    return re.sub(r"\s+", " ", text)


# ==========================================
# Keyword Boosting
# ==========================================
def apply_keyword_boost(text: str, logits):
    text = normalize(text)
    tokens = text.split()

    logits = logits.clone()

    for intent, cfg in KEYWORDS.items():
        idx = label2id[intent]
        for kw in cfg["words"]:
            if kw in text or kw in tokens:
                logits[0][idx] += cfg["boost"]

    return logits


# ==========================================
# related_works 서브 의도 판별 (visual / context)
# ==========================================
def predict_sub_intent_for_related(text: str):
    text = normalize(text)

    visual_kw = ["비슷하게 생긴", "시각", "비주얼", "외형", "색감", "닮은", "겉모습"]
    context_kw = ["주제", "내용", "맥락", "설명", "사조", "시대", "배경"]

    if any(kw in text for kw in visual_kw):
        return "visual"

    if any(kw in text for kw in context_kw):
        return "context"

    # 자동 정답: 시각 기반 추천이 기본값
    return "visual"


# ==========================================
# 최종 Intent 예측 함수
# ==========================================
@torch.no_grad()
def predict_intent(text: str) -> str:
    model = _load_model()
    text = normalize(text)

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=96
    )

    logits = model(
        enc["input_ids"].to(DEVICE),
        enc["attention_mask"].to(DEVICE)
    )

    # Keyword Boosting 적용
    logits = apply_keyword_boost(text, logits)

    # 가장 점수가 높은 intent 선택
    intent_id = logits.argmax(dim=1).item()
    return id2label[intent_id]
