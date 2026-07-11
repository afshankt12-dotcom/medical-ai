import torch
from torch.utils.data import Dataset


class MedicalDataset(Dataset):
    def __init__(self, questions, answers, tokenizer, max_length=64):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def pad_sequence(self, token_ids):
        """
        Pad or truncate token sequence.
        """
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids += [0] * (self.max_length - len(token_ids))

        return token_ids

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]

        question_ids = self.tokenizer.encode(question)
        answer_ids = self.tokenizer.encode(answer)

        question_ids = self.pad_sequence(question_ids)
        answer_ids = self.pad_sequence(answer_ids)

        return {
            "question": torch.tensor(question_ids, dtype=torch.long),
            "answer": torch.tensor(answer_ids, dtype=torch.long),
        }