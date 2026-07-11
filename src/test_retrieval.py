from src.retrieval import Retriever

retriever = Retriever()

while True:

    question = input("\nAsk: ")

    if question.lower() == "exit":
        break

    answer, score = retriever.search(question)

    print("\nConfidence:", round(score, 3))
    print("\nAnswer:\n")
    print(answer)