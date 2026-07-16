import torch

from src.retrieval import Retriever
from src.tokenizer import MedicalTokenizer
from src.transformer_model import MedicalTransformer


MODEL_PATH = "models/medical_transformer.pth"
VOCAB_PATH = "models/vocab.json"


class MedicalChatbot:

    def __init__(self):

        print("Loading Medical Chatbot...")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load tokenizer
        self.tokenizer = MedicalTokenizer()
        self.tokenizer.load_vocab(VOCAB_PATH)

        # Load transformer
        self.model = MedicalTransformer(
            vocab_size=self.tokenizer.vocab_size()
        )

        self.model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=self.device
            )
        )

        self.model.to(self.device)
        self.model.eval()

        # Retrieval system
        self.retriever = Retriever()

        print("Medical Chatbot Ready!")


    # -----------------------------------------
    # Chatbot Answer
    # -----------------------------------------

    def ask(self, question):

        # Search medical database
        answer, score = self.retriever.search(
            question
        )

        return {

            "question": question,

            "answer": answer,

            "confidence": round(score, 3)

        }