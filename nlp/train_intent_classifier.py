import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import autocast, GradScaler



# ----------------------------------------------------------
# 0. 재현성
# ----------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------
# 1. 새 intent_training.csv 로드
# ----------------------------------------------------------
CSV_PATH = "nlp/intent_training.csv"   # ← 새로 생성된 파일
assert os.path.exists(CSV_PATH), f"CSV 없음: {CSV_PATH}"

df = pd.read_csv(CSV_PATH)
assert {"question_ko", "intent"} <= set(df.columns), "CSV에 question_ko, intent 필요"

texts = df["question_ko"].astype(str).tolist()
labels_str = df["intent"].astype(str).tolist()


# ----------------------------------------------------------
# 2. Label Encoding 저장
# ----------------------------------------------------------
le = LabelEncoder()
labels = le.fit_transform(labels_str)
num_labels = len(le.classes_)

with open("nlp/intent_labels.json", "w", encoding="utf-8") as f:
    json.dump(
        {i: c for i, c in enumerate(le.classes_)},
        f, ensure_ascii=False, indent=2
    )


# ----------------------------------------------------------
# 3. Train/Val Split
# ----------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    texts,
    labels,
    test_size=0.1,
    random_state=42,
    stratify=labels
)


# ----------------------------------------------------------
# 4. Tokenizer & Config (로컬 모델 경로!)
# ----------------------------------------------------------
MODEL_NAME = "/Users/jangsunwoo/Documents/ai_docent/ko-sroberta-multitask"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    local_files_only=True
)

MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 5


# ----------------------------------------------------------
# 5. Dataset
# ----------------------------------------------------------
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


train_ds = IntentDataset(X_train, y_train, tokenizer, MAX_LEN)
val_ds   = IntentDataset(X_val,   y_val,   tokenizer, MAX_LEN)

collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)


# ----------------------------------------------------------
# 6. Model
# ----------------------------------------------------------
class IntentClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
            use_safetensors=False
        )
        hidden = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = out.last_hidden_state[:, 0]  # [CLS]
        logits = self.classifier(cls)
        return logits


model = IntentClassifier(num_labels).to(DEVICE)


# ----------------------------------------------------------
# 7. Loss / Optimizer / Scheduler
# ----------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

total_steps = len(train_loader) * EPOCHS
warmup = int(total_steps * 0.1)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    warmup,
    total_steps
)

scaler = GradScaler("cuda", enabled=(DEVICE == "cuda"))


# ----------------------------------------------------------
# 8. Train / Eval
# ----------------------------------------------------------
def train_one_epoch():
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn      = batch["attention_mask"].to(DEVICE)
        labels    = batch["labels"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=(DEVICE == "cuda")):
            logits = model(input_ids, attn)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def eval_one_epoch():
    model.eval()
    correct = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn      = batch["attention_mask"].to(DEVICE)
        labels    = batch["labels"].to(DEVICE)

        logits = model(input_ids, attn)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()

    return correct / len(val_ds)


# ----------------------------------------------------------
# 9. MAIN LOOP
# ----------------------------------------------------------
if __name__ == "__main__":
    best = 0
    bad = 0
    patience = 2

    for ep in range(1, EPOCHS + 1):
        tr = train_one_epoch()
        va = eval_one_epoch()

        print(f"[Epoch {ep}] loss={tr:.4f} | val_acc={va:.4f}")

        if va > best:
            best = va
            bad = 0
            torch.save(model.state_dict(), "nlp/intent_classifier_best.pt")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping")
                break

    torch.save(model.state_dict(), "nlp/intent_classifier.pt")
    print(f"학습 완료 — 최고 정확도: {best:.4f}")
