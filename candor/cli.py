import argparse

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    mem = sub.add_parser("memory")
    mem.add_argument("--questions", required=True)
    mem.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.cmd == "memory":
        print("Would answer", args.questions, "and write to", args.out)

if __name__ == "__main__":
    main()