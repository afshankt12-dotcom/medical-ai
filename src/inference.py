import torch

from src.transformer_model import MedicalTransformer
from src.tokenizer import MedicalTokenizer


class MedicalInference:

    def __init__(self, vocab_size):

        self.tokenizer = MedicalTokenizer()

        self.model = MedicalTransformer(vocab_size=vocab_size)

        self.model.load_state_dict(
            torch.load(
                "models/medical_transformer.pth",
                map_location="cpu"
            )
        )

        self.model.eval()


    def encode(self, text):

        ids = self.tokenizer.encode(text)

        ids = torch.tensor(ids).unsqueeze(0)

        with torch.no_grad():

            embedding = self.model(ids)

        return embedding.mean(dim=1)