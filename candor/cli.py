from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from .answer import IDK, STATS, answer_question, load_env
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

    errors = 0
    with open(out, "w", encoding="utf-8") as f:
        for q in rows:
            try:
                ans = answer_question(
                    q["question"], parse_ts(q["as_of"]), store, index, use_llm=use_llm
                )
            except Exception:
                errors += 1
                traceback.print_exc()
                ans = {"answer": IDK, "sources": [], "retrieved": [], "abstained": True}
            f.write(json.dumps({"id": q["id"], **ans}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{q['id']}: {'ABSTAIN' if ans['abstained'] else ans['answer'][:90]}")

    print(f"\nWrote {len(rows)} answers to {out}")
    print(f"LLM answers: {STATS['llm']} | extractive fallback: {STATS['fallback']} | errors: {errors}")
    if use_llm and STATS["fallback"]:
        print("WARNING: some answers used the fallback (missing key or API errors).")


def main() -> None:
    parser = argparse.ArgumentParser(prog="candor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    memory = sub.add_parser("memory", help="answer a JSONL file of questions")
    memory.add_argument("--questions", required=True)
    memory.add_argument("--out", required=True)
    memory.add_argument("--data", default="data")
    memory.add_argument("--no-llm", action="store_true")

    act = sub.add_parser("actions", help="turn a JSONL file of commands into dry-run actions")
    act.add_argument("--commands", required=True)
    act.add_argument("--out", required=True)
    act.add_argument("--data", default="data")
    act.add_argument("--no-llm", action="store_true")

    rp = sub.add_parser("repl", help="interactive text assistant (dry run)")
    rp.add_argument("--data", default="data")
    rp.add_argument("--as-of", help="ISO time to pretend it is (default: now)")
    rp.add_argument("--no-llm", action="store_true")

    args = parser.parse_args()
    if args.cmd == "memory":
        run(args.questions, args.out, args.data, use_llm=not args.no_llm)
    elif args.cmd == "actions":
        from .actions import run as run_actions
        run_actions(args.commands, args.out, args.data, use_llm=not args.no_llm)
    elif args.cmd == "repl":
        from .actions import repl
        repl(args.data, args.as_of, use_llm=not args.no_llm)


if __name__ == "__main__":
    main()