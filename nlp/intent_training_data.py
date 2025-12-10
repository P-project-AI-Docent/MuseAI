# intent_training_data.py
# -*- coding: utf-8 -*-

import argparse
import random
import re
from typing import List, Dict, Set
import pandas as pd

# ============================
# 1) INTENTS (최종 유지 버전)
# ============================
INTENTS = [
    "artwork_overview",
    "artist_info",
    "date_query",
    "medium_query",
    "metadata_query",
    "related_works",
    "style_context",
    "fallback",
]

# 자연스러운 질문 생성을 위한 랜덤 요소
FILLERS = ["", "", "", "혹시", "그", "일단", "저기", "저", "이", "여기", "저거"]
POLS = ["", "", "좀", "조금", "살짝", "자세히"]
ASK_ENDINGS = ["?", " 알려줘", " 말해줘", " 가르쳐줘", " 알려줄래", " 궁금해"]
HONORIFICS = ["", "요", "요", ""]
PRONOUNS = ["이거", "이 작품", "이건", "이 작품은", "저", "이"]

# 메타데이터 필드 대응
META_FIELDS = {
    "objectID": ["오브젝트 아이디", "번호", "식별자"],
    "title": ["제목", "작품명", "타이틀"],
    "artistDisplayName": ["작가", "예술가", "만든 사람"],
    "objectDate": ["제작 연도", "언제", "몇 년도"],
    "medium": ["재료", "재질", "무엇으로 만들었는지", "소재"],
    "department": ["부서", "컬렉션"],  # metadata_query 일부에서 사용 가능
}

# flatten meta aliases
META_FIELD_ALIAS = [alias for v in META_FIELDS.values() for alias in v]
ARTIST_PLACEHOLDERS = ["작가", "예술가", "만든 사람", "그린 사람"]

# ============================
# 2) Intent별 템플릿
# ============================
SEED_TEMPLATES = {
    "artwork_overview": [
        "{f}{p} {pr} 전체적으로 어떤 작품인지{e}",
        "{f}{p} {pr} 전반적으로{e}",
        "{f}{p} {pr} 대략적으로 어떤 작품인지{e}",
        "{f}{p} {pr} 개요 좀{e}",
        "{f}{p} {pr} 소개해줘{h}",
        "{f}{p} {pr} 어떤 내용이야{e}",
        "{f}{p} {pr} 작품에 대해 알려줘{e}",
    ],
    "artist_info": [
        "{f}{p} 이거 {a}가 누군지{e}",
        "{f}{p} {a}에 대해 알려줘{h}",
        "{f}{p} 만든 사람 어떤 사람이야{e}",
        "{f}{p} 그린 사람 어떤 사람이야{e}",
        "{f}{p} 작가 배경 설명해줘{h}",
        "{f}{p} 작가 설명해줘{h}",
    ],
    "date_query": [
        "{f}{p} {pr} 언제 만들어졌어{h}",
        "{f}{p} {pr} 제작 연도 알려줘{e}",
        "{f}{p} {pr} 몇 년도 작품이야{e}",
        "{f}{p} {pr} 무슨 시대에 만들어졌어{e}",
        "{f}{p} {pr} 시기 알려줘{e}",
        "{f}{p} {pr} 년도 알려줘{e}",
    ],
    "medium_query": [
        "{f}{p} {pr} 재료가 뭐야{e}",
        "{f}{p} {pr} 어떤 재료 사용했어{e}",
        "{f}{p} {pr} 소재 알려줘{h}",
        "{f}{p} {pr} 질감 알려줘{h}",
        "{f}{p} {pr} 재료 알려줘{h}",
    ],
    "metadata_query": [
        "{f}{p} {pr} {mf} 알려줘{h}",
        "{f}{p} {mf}만 알려줄래{h}",
        "{f}{p} {pr} {mf} 뭐야{e}",
    ],
    "related_works": [
        "{f}{p} {pr} 비슷한 작품 추천해줘{h}",
        "{f}{p} {pr} 유사 작품 있어{e}",
        "{f}{p} {pr} 관련 작품 더 보여줘{e}",
    ],
    "style_context": [
        "{f}{p} {pr} 스타일이 어떤 느낌이야{e}",
        "{f}{p} {pr} 양식적으로 어떤 특징 있어{e}",
        "{f}{p} {pr} 기법적 특징 설명해줘{h}",
        "{f}{p} {pr} 화풍 설명해줘{h}",
        "{f}{p} {pr} 배경 설명해줘{h}",
        "{f}{p} {pr} 사조 설명해줘{h}",
        "{f}{p} {pr} 미술적 특징 설명해줘{h}",
    ],
    "fallback": [
        "{f} 다시 말해줄래{h}",
        "{f} 잘 이해 못했어{h}",
        "{f} 조금 더 구체적으로 말해줘{h}",
        "{f} 더 설명해줘{h}",
    ],
}

# ============================
# 3) 패러프레이즈 규칙
# ============================
SYNONYM_SWAPS = [
    ("알려줘", "말해줘"),
    ("말해줘", "가르쳐줘"),
    ("있나", "있을까"),
    ("있니", "있어"),
]

ENDINGS = ["?", "", " 알려줘", " 말해줘", " 설명해줘"]
TONE_PREFIX = ["", "혹시 ", "음 ", "그 ", "일단 "]

def clean_spaces(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace("요요", "요").replace("??", "?")

def render_seed(intent: str) -> str:
    tpl = random.choice(SEED_TEMPLATES[intent])
    f = random.choice(FILLERS)
    p = (" " + random.choice(POLS)) if random.choice(POLS) else ""
    pr = random.choice(PRONOUNS)
    a = random.choice(ARTIST_PLACEHOLDERS)
    e = random.choice(ASK_ENDINGS)
    h = random.choice(HONORIFICS)
    mf = random.choice(META_FIELD_ALIAS)
    s = tpl.format(f=f, p=p, pr=pr, a=a, e=e, h=h, mf=mf)
    return clean_spaces(s)

def paraphrase_once(s: str) -> str:
    t = random.choice(TONE_PREFIX) + s
    for _ in range(random.randint(1, 2)):
        a, b = random.choice(SYNONYM_SWAPS)
        t = t.replace(a, b)
    t = t.rstrip("?").strip() + random.choice(ENDINGS)
    return clean_spaces(t)

def generate_for_intent(intent: str, seeds_per_intent: int, pmin: int, pmax: int):
    uniq = set()
    seeds = []

    tries = seeds_per_intent * 10
    while len(seeds) < seeds_per_intent and tries > 0:
        q = render_seed(intent)
        if q not in uniq:
            uniq.add(q)
            seeds.append(q)
        tries -= 1

    final = []
    seen = set()

    for q in seeds:
        final.append(q)
        seen.add(q)
        k = random.randint(pmin, pmax)
        for _ in range(k):
            cand = paraphrase_once(q)
            if cand not in seen:
                seen.add(cand)
                final.append(cand)

    return final

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--seeds_per_intent", type=int, default=80)
    ap.add_argument("--pmin", type=int, default=2)
    ap.add_argument("--pmax", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    rows = []
    for intent in INTENTS:
        qs = generate_for_intent(intent, args.seeds_per_intent, args.pmin, args.pmax)
        for q in qs:
            rows.append({"question_ko": q, "intent": intent})

    df = pd.DataFrame(rows)
    df = df.drop_duplicates().sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    print(f"[DONE] Total rows: {len(df)}")

if __name__ == "__main__":
    main()
