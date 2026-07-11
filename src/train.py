import os

import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import Dataset, DataLoader

from src.tokenizer import MedicalTokenizer
from src.transformer_model import MedicalTransformer


# ==========================================
# Configuration
# ==========================================

DATA_PATH = "dataset/medquad_hf.parquet"

MODEL_PATH = "models/medical_transformer.pth"

VOCAB_PATH = "models/vocab.json"

MAX_LENGTH = 64

BATCH_SIZE = 16

EPOCHS = 5

LEARNING_RATE = 5e-4


# ==========================================
# Dataset Class
# ==========================================

class MedicalDataset(Dataset):

    def __init__(self, texts, tokenizer):

        self.samples = []

        for text in texts:

            tokens = tokenizer.encode(
                text,
                max_length=MAX_LENGTH
            )

            input_ids = tokens[:-1]

            target_ids = tokens[1:]

            self.samples.append(
                (
                    torch.tensor(
                        input_ids,
                        dtype=torch.long
                    ),
                    torch.tensor(
                        target_ids,
                        dtype=torch.long
                    )
                )
            )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        return self.samples[index]
    # ==========================================
# Load Dataset
# ==========================================

print("Loading dataset...")

df = pd.read_parquet(DATA_PATH)

# Remove rows with missing values
df = df.dropna(
    subset=[
        "question",
        "answer"
    ]
)

questions = df["question"].astype(str).tolist()

answers = df["answer"].astype(str).tolist()

texts = []

for question, answer in zip(questions, answers):

    texts.append(
        question + " " + answer
    )

print(f"Loaded samples: {len(texts)}")


# ==========================================
# Tokenizer
# ==========================================

tokenizer = MedicalTokenizer()

tokenizer.build_vocab(texts)

print(f"Vocabulary Size: {tokenizer.vocab_size()}")

os.makedirs(
    "models",
    exist_ok=True
)

tokenizer.save_vocab(VOCAB_PATH)


# ==========================================
# DataLoader
# ==========================================

dataset = MedicalDataset(
    texts,
    tokenizer
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(f"Training Samples: {len(dataset)}")


# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")


# ==========================================
# Model
# ==========================================

model = MedicalTransformer(
    vocab_size=tokenizer.vocab_size(),
    max_len=MAX_LENGTH
)

model = model.to(device)

print("Model initialized successfully.")
# ==========================================
# Loss Function
# ==========================================

criterion = nn.CrossEntropyLoss(
    ignore_index=tokenizer.word2idx["<PAD>"]
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=2,
    gamma=0.5
)

print("\nStarting Training...\n")


# ==========================================
# Training Loop
# ==========================================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for input_ids, target_ids in loader:

        input_ids = input_ids.to(device)

        target_ids = target_ids.to(device)

        optimizer.zero_grad()

        logits, sentence_embedding = model(input_ids)

        loss = criterion(

            logits.reshape(
                -1,
                tokenizer.vocab_size()
            ),

            target_ids.reshape(-1)

        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    avg_loss = total_loss / len(loader)

    current_lr = scheduler.get_last_lr()[0]

    print(
        f"Epoch {epoch+1}/{EPOCHS}"
        f" | Loss: {avg_loss:.4f}"
        f" | LR: {current_lr:.6f}"
    )


# ==========================================
# Save Model
# ==========================================

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("\nTraining Complete!")

print("Model saved to:", MODEL_PATH)

print("Vocabulary saved to:", VOCAB_PATH)