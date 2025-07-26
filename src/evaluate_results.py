# src/evaluate_results.py

import json
import matplotlib.pyplot as plt
import argparse
import os


def load_history(path):
    """Load results from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return data["strategy"], data["results"]


def plot_learning_curves(histories, output_path):
    """
    Plot learning curves from multiple strategies.

    Args:
        histories (list): List of (strategy_name, result_list)
        output_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))

    for strategy, history in histories:
        steps, accs, f1s = zip(*history)
        plt.plot(steps, accs, label=f"{strategy} - Accuracy", linestyle="--")
        plt.plot(steps, f1s, label=f"{strategy} - F1 Score", linestyle="-")

    plt.xlabel("Number of Labeled Samples")
    plt.ylabel("Score")
    plt.title("Active Learning Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Paths to JSON history files (e.g., history_entropy.json history_random.json)"
    )
    parser.add_argument(
        "--output",
        default="results/learning_curve.png",
        help="Path to save the learning curve plot"
    )
    args = parser.parse_args()

    histories = []
    for path in args.inputs:
        strategy, results = load_history(path)
        histories.append((strategy, results))

    plot_learning_curves(histories, args.output)


if __name__ == "__main__":
    main()
