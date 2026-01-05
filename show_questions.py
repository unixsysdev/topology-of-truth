"""Quick script to preview questions from the LIMO dataset."""

from datasets import load_dataset
import argparse


def main():
    parser = argparse.ArgumentParser(description="Preview LIMO dataset questions")
    parser.add_argument("-n", "--num", type=int, default=5, help="Number of questions to show")
    parser.add_argument("--start", type=int, default=0, help="Starting index")
    parser.add_argument("--full", action="store_true", help="Show full question (no truncation)")
    args = parser.parse_args()

    print("Loading LIMO dataset...")
    dataset = load_dataset("GAIR/LIMO", split="train")
    print(f"Total questions: {len(dataset)}\n")

    for i in range(args.start, min(args.start + args.num, len(dataset))):
        item = dataset[i]
        question = item.get("question", item.get("problem", ""))
        answer = item.get("answer", item.get("solution", ""))

        print(f"{'='*60}")
        print(f"Question {i}")
        print(f"{'='*60}")
        
        if args.full or len(question) < 500:
            print(question)
        else:
            print(question[:500] + "...")
        
        print(f"\n--- Answer ---")
        if args.full or len(answer) < 200:
            print(answer)
        else:
            print(answer[:200] + "...")
        print()


if __name__ == "__main__":
    main()
