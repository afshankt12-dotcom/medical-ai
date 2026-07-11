import torch

from src.transformer_model import MedicalTransformer

VOCAB_SIZE = 5000

model = MedicalTransformer(vocab_size=VOCAB_SIZE)

x = torch.randint(0, VOCAB_SIZE, (2, 20))

logits, embedding = model(x)

print("Input Shape      :", x.shape)
print("Logits Shape     :", logits.shape)
print("Embedding Shape  :", embedding.shape)