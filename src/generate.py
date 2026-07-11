import torch

from src.tokenizer import MedicalTokenizer
from src.transformer_model import MedicalTransformer


MODEL_PATH = "models/medical_transformer.pth"
VOCAB_PATH = "models/vocab.json"

MAX_LENGTH = 64


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# Load tokenizer

tokenizer = MedicalTokenizer()

tokenizer.load_vocab(
    VOCAB_PATH
)


# Load model

model = MedicalTransformer(
    vocab_size=tokenizer.vocab_size()
)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


model.to(device)

model.eval()



def generate_answer(question):

    tokens = tokenizer.encode(
        question,
        MAX_LENGTH
    )


    input_ids = torch.tensor(
        tokens[:-1]
    ).unsqueeze(0)


    input_ids = input_ids.to(device)


    with torch.no_grad():

        output = model(
            input_ids
        )


    predicted = torch.argmax(
        output,
        dim=-1
    )


    words = tokenizer.decode(
        predicted[0].cpu().tolist()
    )


    return words



if __name__ == "__main__":

    while True:

        question = input(
            "\nYou: "
        )


        if question.lower() == "exit":
            break


        answer = generate_answer(
            question
        )


        print(
            "\nAI:",
            answer
        )   