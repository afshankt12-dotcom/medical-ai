from src.chatbot import MedicalChatbot


def main():

    chatbot = MedicalChatbot()

    print("\nMedical AI Chatbot")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            break

        result = chatbot.ask(question)

        print("\nConfidence:", result["confidence"])
        print("\nAnswer:\n")
        print(result["answer"])
        print("-" * 80)


if __name__ == "__main__":
    main()