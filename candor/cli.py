from __future__ import annotations

import argparse
import json
from pathlib import Path

from .answer import STATS, answer_question, load_env
from .load import load_all, parse_ts
from .retrieve import Index


def run(questions: str, out: str, data: str = "data", use_llm: bool = True) -> None:
    load_env()
    store = load_all(data)
    index = Index(store)

    rows = [
        json.loads(l)
        for l in Path(questions).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for q in rows:
            ans = answer_question(
                q["question"],
                parse_ts(q["as_of"]),
                store,
                index,
                use_llm=use_llm,
            )
            f.write(
                json.dumps(
                    {"id": q["id"], **ans},
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(
                f"{q['id']}: "
                f"{'ABSTAIN' if ans['abstained'] else ans['answer'][:90]}"
            )

    print(f"\nWrote {len(rows)} answers to {out}")
    print(
        f"LLM answers: {STATS['llm']} | "
        f"extractive fallback: {STATS['fallback']}"
    )
    if use_llm and STATS["fallback"]:
        print(
            "WARNING: some answers used the fallback "
            "(missing key or API errors)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="candor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    memory = sub.add_parser(
        "memory",
        help="answer a JSONL file of questions",
    )
    memory.add_argument("--questions", required=True)
    memory.add_argument("--out", required=True)
    memory.add_argument("--data", default="data")
    memory.add_argument("--no-llm", action="store_true")

    args = parser.parse_args()

    if args.cmd == "memory":
        run(
            args.questions,
            args.out,
            args.data,
            use_llm=not args.no_llm,
        )


if __name__ == "__main__":
    main()