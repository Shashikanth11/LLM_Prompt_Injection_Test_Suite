import argparse
from dataset_loader import load_and_prepare_dataset
from runners import run_suite_on_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3")
    parser.add_argument("--dataset", default="data/ecomm_attack_suite.csv")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    dataset_df = load_and_prepare_dataset(args.dataset)
    result = run_suite_on_dataset(args.model, dataset_df, temperature=args.temperature, max_examples=args.max_examples)

    print(f"\n✅ Evaluation complete.")
    print(f"CSV results: {result['csv']}")
    print(f"Metrics JSON: {result['summary']}")
    print(f"Plots saved: {result['plots']}")
    print("\n=== Metrics Summary ===")
    for k,v in result['metrics'].items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
