"""
CLIP Vision-only LoRA Fine-tuning (for 1000-image dataset)
cv_clip/clip_fine_tuning.py
"""

import os
from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    CLIPModel,
    CLIPProcessor,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
from torchvision import transforms
import random
import numpy as np

# LoRA
from peft import LoraConfig, get_peft_model


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
CSV_PATH = "./met20k/metadata_clip_text.csv"
IMAGE_ROOT = "./met20k/images"

# Titan 서버 로컬 모델
MODEL_NAME = "/home/cvip-titan/sunwoo/ai_docent/clip_base"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 3
LR = 1e-5
IMG_SIZE = 224   # CLIP 기본 입력 크기


# ---------------------------------------------------------
# SEED
# ---------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed()


# ---------------------------------------------------------
# AUGMENTATION
# ---------------------------------------------------------
AUG = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.RandomRotation(10),
])


# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------
class MetDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.root = Path(IMAGE_ROOT)

    def __len__(self):
        return len(self.df)

    def safe_path(self, p: str):
        p = Path(p)
        return p if p.is_absolute() else (self.root / p)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.safe_path(row["localImagePath"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")

        image = AUG(image)

        text = row.get("clip_text", "")
        if not isinstance(text, str) or len(text.strip()) == 0:
            text = "An artwork from the Metropolitan Museum of Art."

        return image, text


# ---------------------------------------------------------
# TRAINING LOOP (LoRA)
# ---------------------------------------------------------
def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["localImagePath"]).reset_index(drop=True)

    dataset = MetDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: x,
    )

    print("Loading base CLIP...")

    model = CLIPModel.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
        use_safetensors=False
    ).to(DEVICE)

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME,
        local_files_only=True
    )

    # ---------------------------------------------------------
    # LoRA CONFIG (Vision encoder only)
    # ---------------------------------------------------------
    print("Applying LoRA adapters...")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.01
    )

    total_steps = len(loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(50, total_steps // 10),
        num_training_steps=total_steps,
    )

    print("Start training...")
    model.train()

    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            try:
                images = [img for img, _ in batch]
                texts = [txt for _, txt in batch]

                inputs = processor(
                    images=images,
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77,
                ).to(DEVICE)

                out = model(**inputs, return_loss=True)
                loss = out.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                pbar.set_postfix(loss=float(loss.detach().cpu()))

            except Exception as e:
                print("Batch skipped due to error:", e)
                continue

    # ---------------------------------------------------------
    # SAVE LoRA MODEL
    # ---------------------------------------------------------
    save_dir = Path("./clip_lora")
    save_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    print("\n🔥 LoRA Fine-tuned CLIP saved to:", save_dir)


if __name__ == "__main__":
    main()
