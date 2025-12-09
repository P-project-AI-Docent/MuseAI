"""
Zero-shot CLIP vs LoRA-Finetuned CLIP
Image Retrieval Evaluation Script (Multi-trial)

- Top-1 Accuracy
- Recall@K
- Mean Cosine Similarity
- 200장 무작위 샘플 × N회 반복 → 평균, 표준편차
"""

import os
import json
import csv
import random
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from torchvision import transforms
import faiss

from transformers import CLIPModel, CLIPProcessor
from peft import PeftModel

# 경로 설정
BASE_CLIP_DIR = "/home/cvip-titan/sunwoo/ai_docent/clip_base"

ZERO_INDEX_DIR = "./index_assets"          # Zero-shot: paths.json = list[str]
LORA_INDEX_DIR = "./index_assets_lora"     # LoRA: paths.json = list[dict]

LORA_DIR = "./clip_lora"                   # LoRA 어댑터 폴더

TEST_IMAGE_DIR = "./met20k/images"         # 평가용 이미지 원본

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 유틸
def list_images(folder: str) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted([p for p in Path(folder).glob("*") if p.suffix.lower() in exts])


def safe_open_image(path: Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return Image.new("RGB", (224, 224), "white")


# 인덱스 로드
def load_index(index_dir: str):
    index = faiss.read_index(str(Path(index_dir) / "images.faiss"))
    with open(Path(index_dir) / "paths.json", "r", encoding="utf-8") as f:
        paths = json.load(f)
    return index, paths


# 쿼리 증강
def make_query_augment(severity=0):
    if severity == 0 or severity is None:
        return None
    if severity == 1:
        ops = [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        ]
    elif severity == 2:
        ops = [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
        ]
    else:
        ops = [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.15),
            transforms.GaussianBlur(kernel_size=7, sigma=(0.5, 2.0)),
        ]
    return transforms.Compose(ops)


# 임베딩
@torch.no_grad()
def embed(model, processor, img: Image.Image) -> np.ndarray:
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    feat = model.get_image_features(**inputs)
    feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")


# 테스트 샘플링
def sample_test_images(folder: str, n=200) -> List[Tuple[str, int]]:
    paths = list_images(folder)
    if len(paths) == 0:
        raise RuntimeError(f"No test images found in {folder}")
    n = min(n, len(paths))
    sample = random.sample(paths, n)
    return [(str(p), int(Path(p).stem)) for p in sample]


# 단일 평가
def evaluate(model, processor, index, paths, test_items, topk=5, aug_severity=0) -> Dict[str, float]:
    aug = make_query_augment(aug_severity)
    total = 0
    top1_correct = 0
    recallk_correct = 0
    sims = []

    for test_path, true_oid in tqdm(test_items, desc="Evaluating", leave=False):
        img = safe_open_image(Path(test_path))
        if aug is not None:
            img = aug(img)

        q = embed(model, processor, img)
        D, I = index.search(q, topk)

        sims.append(float(D[0][0]))

        retrieved_oids = []
        for idx in I[0]:
            entry = paths[idx]
            if isinstance(entry, dict):              # LoRA index
                oid = entry.get("objectID")
            else:                                    # Zero-shot index
                try:
                    oid = int(Path(entry).stem)
                except Exception:
                    oid = None
            retrieved_oids.append(oid)

        total += 1
        if retrieved_oids and retrieved_oids[0] == true_oid:
            top1_correct += 1
        if true_oid in retrieved_oids[:topk]:
            recallk_correct += 1

    top1 = top1_correct / total if total else 0.0
    recallk = recallk_correct / total if total else 0.0
    mean_sim = (sum(sims) / len(sims)) if sims else 0.0

    return {"top1": top1, "recallk": recallk, "mean_sim": mean_sim, "total": total}


# 두 모델 로드
def load_models():
    zero_model = CLIPModel.from_pretrained(BASE_CLIP_DIR).to(DEVICE).eval()
    zero_proc = CLIPProcessor.from_pretrained(BASE_CLIP_DIR)
    zero_index, zero_paths = load_index(ZERO_INDEX_DIR)

    base_model = CLIPModel.from_pretrained(BASE_CLIP_DIR).to(DEVICE)
    lora_model = PeftModel.from_pretrained(base_model, LORA_DIR, is_trainable=False).to(DEVICE).eval()
    lora_proc = CLIPProcessor.from_pretrained(BASE_CLIP_DIR)
    lora_index, lora_paths = load_index(LORA_INDEX_DIR)
    return (zero_model, zero_proc, zero_index, zero_paths), (lora_model, lora_proc, lora_index, lora_paths)


# 한 번의 비교 평가
def evaluate_once(topk=5, aug=0, sample_n=200, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    test_items = sample_test_images(TEST_IMAGE_DIR, n=sample_n)

    (zero_m, zero_p, zero_i, zero_paths), (lora_m, lora_p, lora_i, lora_paths) = load_models()

    res_zero = evaluate(zero_m, zero_p, zero_i, zero_paths, test_items, topk=topk, aug_severity=aug)
    res_lora = evaluate(lora_m, lora_p, lora_i, lora_paths, test_items, topk=topk, aug_severity=aug)
    return res_zero, res_lora


# 다회 반복 평가
def evaluate_multi(trials=3, topk=5, aug=0, sample_n=200, base_seed=42, csv_out=None):
    zero_top1, zero_rk, zero_sim = [], [], []
    lora_top1, lora_rk, lora_sim = [], [], []

    for t in range(trials):
        seed = base_seed + t
        print(f"\n[Trial {t+1}/{trials}] n={sample_n}, topk={topk}, aug={aug}, seed={seed}")
        res_zero, res_lora = evaluate_once(topk=topk, aug=aug, sample_n=sample_n, seed=seed)

        print(f"  Zero-shot  Top-1={res_zero['top1']:.4f}  R@{topk}={res_zero['recallk']:.4f}  MeanSim={res_zero['mean_sim']:.4f}")
        print(f"  LoRA       Top-1={res_lora['top1']:.4f}  R@{topk}={res_lora['recallk']:.4f}  MeanSim={res_lora['mean_sim']:.4f}")

        zero_top1.append(res_zero["top1"]); zero_rk.append(res_zero["recallk"]); zero_sim.append(res_zero["mean_sim"])
        lora_top1.append(res_lora["top1"]); lora_rk.append(res_lora["recallk"]); lora_sim.append(res_lora["mean_sim"])

    def m_s(a):  # 평균, 표준편차
        return float(np.mean(a)), float(np.std(a))

    z_top1_m, z_top1_s = m_s(zero_top1)
    z_rk_m,   z_rk_s   = m_s(zero_rk)
    z_sim_m,  z_sim_s  = m_s(zero_sim)

    l_top1_m, l_top1_s = m_s(lora_top1)
    l_rk_m,   l_rk_s   = m_s(lora_rk)
    l_sim_m,  l_sim_s  = m_s(lora_sim)

    print("\n============================")
    print(" MULTI-TRIAL COMPARISON ")
    print("============================")
    print(f"Trials           : {trials}")
    print(f"Samples per trial: {sample_n}")
    print("----------------------------------------")
    print(f"Zero-shot Top-1   mean={z_top1_m:.4f}  std={z_top1_s:.4f}")
    print(f"LoRA     Top-1    mean={l_top1_m:.4f}  std={l_top1_s:.4f}")
    print("----------------------------------------")
    print(f"Zero-shot R@{topk}  mean={z_rk_m:.4f}  std={z_rk_s:.4f}")
    print(f"LoRA     R@{topk}   mean={l_rk_m:.4f}  std={l_rk_s:.4f}")
    print("----------------------------------------")
    print(f"Zero-shot MeanSim mean={z_sim_m:.4f}  std={z_sim_s:.4f}")
    print(f"LoRA     MeanSim  mean={l_sim_m:.4f}  std={l_sim_s:.4f}")
    print("============================")

    if csv_out:
        Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "zero_mean", "zero_std", "lora_mean", "lora_std", "trials", "n", "topk", "aug"])
            w.writerow(["top1", z_top1_m, z_top1_s, l_top1_m, l_top1_s, trials, sample_n, topk, aug])
            w.writerow(["recall_at_k", z_rk_m, z_rk_s, l_rk_m, l_rk_s, trials, sample_n, topk, aug])
            w.writerow(["mean_similarity", z_sim_m, z_sim_s, l_sim_m, l_sim_s, trials, sample_n, topk, aug])
        print(f"\nCSV 저장 완료: {csv_out}")

    # 리포트에서 쓰기 편하도록 dict 반환
    return {
        "zero": {"top1": (z_top1_m, z_top1_s), "recallk": (z_rk_m, z_rk_s), "meansim": (z_sim_m, z_sim_s)},
        "lora": {"top1": (l_top1_m, l_top1_s), "recallk": (l_rk_m, l_rk_s), "meansim": (l_sim_m, l_sim_s)},
        "trials": trials,
        "n": sample_n,
        "topk": topk,
        "aug": aug,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3, help="반복 횟수")
    parser.add_argument("--n", type=int, default=200, help="각 반복의 샘플 수")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--aug", type=int, default=0, help="증강 수준 0/1/2/3")
    parser.add_argument("--csv", type=str, default="", help="결과 CSV 저장 경로")
    args = parser.parse_args()

    # 반복 평가 실행
    evaluate_multi(
        trials=args.trials,
        topk=args.topk,
        aug=args.aug,
        sample_n=args.n,
        base_seed=42,
        csv_out=(args.csv if args.csv else None),
    )
