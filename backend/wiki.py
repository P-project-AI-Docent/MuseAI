# backend/wiki.py
import requests
import ollama
import urllib.parse
import re
from typing import Optional, List, Tuple

TIMEOUT = 8
UA = {"User-Agent": "Mozilla/5.0"}

# ============================================================
# 작가 이름 정제
# ============================================================
def _clean_artist_name(name: str) -> str:
    name = re.sub(r"\(.*?\)", "", name)
    return name.strip()


# ============================================================
# 1) Wikidata 검색 → QID 얻기
# ============================================================
def _wikidata_search(query: str) -> Optional[str]:
    try:
        url = (
            "https://www.wikidata.org/w/api.php"
            "?action=wbsearchentities"
            f"&search={urllib.parse.quote(query)}"
            "&language=en"
            "&format=json"
        )
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        data = r.json()

        results = data.get("search")
        if not results:
            return None

        return results[0].get("id")
    except:
        return None


# ============================================================
# 2) QID → ko 위키 제목(kowiki title)
# ============================================================
def _wikidata_get_korean_title(qid: str) -> Optional[str]:
    try:
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        entity_data = r.json()

        entities = entity_data.get("entities", {})
        entity = entities.get(qid, {})
        sitelinks = entity.get("sitelinks", {})

        if "kowiki" in sitelinks:
            return sitelinks["kowiki"].get("title")

        labels = entity.get("labels", {})
        if "ko" in labels:
            return labels["ko"].get("value")

        return None
    except:
        return None


# ============================================================
# 3) Wikipedia Summary 가져오기 (ko/en)
# ============================================================
def _fetch_summary_from_wiki(title: str, lang: str) -> Optional[str]:
    try:
        encoded = urllib.parse.quote(title.replace(" ", "_"))
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        r = requests.get(url, headers=UA, timeout=TIMEOUT)

        if r.status_code != 200:
            return None

        data = r.json()
        extract = (data.get("extract") or "").strip()
        if not extract:
            return None

        sents = extract.split(".")
        trimmed = ".".join(sents[:2]).strip()
        if trimmed and not trimmed.endswith("."):
            trimmed += "."

        return trimmed

    except:
        return None


# ============================================================
# 4) 영어 문장 한국어 번역
# ============================================================
def _translate_to_korean(text: str) -> str:
    prompt = f"""
    다음 문장을 자연스럽고 간결한 한국어 1~2문장으로 번역하세요.

    [절대 규칙]
    - 영어 알파벳 포함 금지
    - 외국어 금지
    - 한자 금지
    - 중복 금지

    원문:
    {text}
    """

    try:
        resp = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        return resp["message"]["content"].strip()

    except:
        return ""


# ============================================================
# 5) 최종 wiki 요약 얻기
# ============================================================
def wiki_summary_best_effort(queries: List[str]) -> Tuple[Optional[str], str]:

    for q in queries:
        if not q:
            continue

        # 1) Wikidata 기반 검색
        qid = _wikidata_search(q)
        if qid:
            ko_title = _wikidata_get_korean_title(qid)
            if ko_title:
                txt_ko = _fetch_summary_from_wiki(ko_title, "ko")
                if txt_ko:
                    return txt_ko, "ko-wikidata"

                txt_en = _fetch_summary_from_wiki(ko_title, "en")
                if txt_en:
                    tr = _translate_to_korean(txt_en)
                    return tr or None, "en-translated-wikidata"

        # 2) direct ko
        txt_ko2 = _fetch_summary_from_wiki(q, "ko")
        if txt_ko2:
            return txt_ko2, "ko"

        # 3) direct en
        txt_en2 = _fetch_summary_from_wiki(q, "en")
        if txt_en2:
            tr2 = _translate_to_korean(txt_en2)
            return tr2 or None, "en-translated"

    return None, ""
