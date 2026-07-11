from src.chatbot import MedicalChatbot
import csv

QUESTIONS = [
    "What is diabetes?",
    "What are the symptoms of malaria?",
    "What causes asthma?",
    "What is hypertension?",
    "How can I prevent heart disease?",
    "What is anemia?",
    "What causes fever?",
    "What is pneumonia?",
    "What are the symptoms of dengue?",
    "What is COVID-19?",
    "What causes kidney stones?",
    "What is arthritis?",
    "What are the symptoms of tuberculosis?",
    "What is obesity?",
    "What causes migraine?",
    "What is hepatitis?",
    "What is thyroid disease?",
    "What is cholesterol?",
    "What is a healthy diet?",
    "When should I see a doctor for chest pain?"
]


def main():

    chatbot = MedicalChatbot()

    with open(
        "evaluation_results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Question",
            "Answer",
            "Confidence"
        ])

        print("\nMODEL EVALUATION\n")

        for i, question in enumerate(QUESTIONS, start=1):

            result = chatbot.ask(question)

            print(f"{i}. {question}")
            print(f"Confidence : {result['confidence']}")
            print(f"Answer      : {result['answer']}")
            print("-" * 80)

            writer.writerow([
                question,
                result["answer"],
                result["confidence"]
            ])

    print("\nEvaluation completed.")
    print("Results saved to evaluation_results.csv")


if __name__ == "__main__":
    main()