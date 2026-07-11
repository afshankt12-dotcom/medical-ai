import torch

from deep_translator import GoogleTranslator
from langdetect import detect

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
    # Detect Language
    # -----------------------------------------

    def detect_user_language(self, text):

        manglish_words = [
            "enikku",
            "enik",
            "pani",
            "undu",
            "illa",
            "aano",
            "entha",
            "engane",
            "vedana",
            "thalavedana",
            "marunnu",
            "asukham",
            "sukham",
            "vayar"
        ]

        text_lower = text.lower()

        # Manglish detection
        for word in manglish_words:
            if word in text_lower:
                return "ml"

        # Normal detection
        try:
            return detect(text)

        except Exception:
            return "en"



    # -----------------------------------------
    # Manglish Converter
    # -----------------------------------------

    def convert_manglish_to_malayalam(self, text):

        manglish_map = {

            "enikku": "എനിക്ക്",
            "enik": "എനിക്ക്",

            "pani": "പനി",
            "undu": "ഉണ്ട്",
            "illa": "ഇല്ല",

            "thalavedana": "തലവേദന",
            "vedana": "വേദന",

            "marunnu": "മരുന്ന്",
            "asukham": "അസുഖം",

            "vayar": "വയർ",
            "sukham": "സുഖം",

            "alla": "അല്ല"

        }


        words = text.lower().split()

        result = []


        for word in words:

            if word in manglish_map:
                result.append(
                    manglish_map[word]
                )

            else:
                result.append(word)


        return " ".join(result)



    # -----------------------------------------
    # Translation
    # -----------------------------------------

    def translate_text(self, text, target):

        try:

            return GoogleTranslator(
                source="auto",
                target=target
            ).translate(text)

        except Exception:

            return text



    # -----------------------------------------
    # Chatbot Answer
    # -----------------------------------------

    def ask(self, question):

        original_question = question


        # Detect language
        user_language = self.detect_user_language(
            question
        )


        # Convert Manglish first
        if user_language == "ml":

            question = self.convert_manglish_to_malayalam(
                question
            )


        # Convert to English for medical retrieval
        english_question = self.translate_text(
            question,
            "en"
        )


        # Search medical database
        answer, score = self.retriever.search(
            english_question
        )


        # Translate answer back
        if user_language != "en":

            answer = self.translate_text(
                answer,
                user_language
            )


        return {

            "question": original_question,

            "answer": answer,

            "confidence": round(score, 3)

        }