import pandas as pd
from transformers import CLIPTokenizerFast
from tqdm import tqdm

CSV_IN = "met20k/metadata_with_description.csv"      # input CSV
CSV_OUT = "met20k/metadata_clip_text.csv"            # output CSV

# CLIP tokenizer (ViT-B/32 기준)
tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

MAX_TOKENS = 77

def smart_truncate(text, max_tokens=MAX_TOKENS):
    """
    CLIP 최대 토큰(77)에 맞게 텍스트를 자르는 함수
    문장을 최대한 손상 없이 자르며,
    토크나이저 기준으로 safe truncate.
    """
    tokens = tokenizer.encode(text, truncation=False)
    if len(tokens) <= max_tokens:
        return text

    # 토큰을 자르고 다시 디코딩
    truncated = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
    return truncated.strip()


def build_clip_text(row):
    """
    설명 + 제목 + 작가로 문장을 구성
    """
    title = str(row.get("title", "")).strip()
    artist = str(row.get("artistDisplayName", "")).strip()
    desc = str(row.get("description", "")).strip()

    # description이 없으면 title/artist 대체
    if not desc:
        desc = ""

    # 조합 (너가 요청한 구조)
    combined = f"{desc} Title: {title}. Artist: {artist}."

    # CLIP 토큰 길이 제한 적용
    combined = smart_truncate(combined)

    return combined


def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_IN)

    print("Generating clip_text column...")

    clip_texts = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        clip_texts.append(build_clip_text(row))

    df["clip_text"] = clip_texts

    df.to_csv(CSV_OUT, index=False)
    print(f"\n완료! 저장 위치 → {CSV_OUT}")
    print("이제 파인튜닝 시 이 clip_text 컬럼만 사용하면 됩니다.")


if __name__ == "__main__":
    main()
