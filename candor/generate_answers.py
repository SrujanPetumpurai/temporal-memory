import json

from candor.load import load_all, parse_ts
from candor.retrieve import Index


def main():
    store = load_all("data")
    index = Index(store)

    with open("evals/memory_train.jsonl", "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    with open("outputs/your_answers.jsonl", "w", encoding="utf-8") as f:
        for q in questions:
            result = index.search(
                q["question"],
                parse_ts(q["as_of"]),
                k=20,
            )

            output = {
                "id": q["id"],
                "answer": "",
                "sources": [],
                "retrieved": result.ids,
                "abstained": result.abstain,
            }

            f.write(json.dumps(output) + "\n")

    print(f"Wrote {len(questions)} answers to outputs/your_answers.jsonl")


if __name__ == "__main__":
    main()