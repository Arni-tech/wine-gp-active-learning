# scripts/run_active_learning.py

import os
import json
from preprocessing import load_and_preprocess_data, load_config
from active_learning import active_learning_loop

def run_and_save(strategy_name):
    print(f"📦 Loading config and data for strategy: {strategy_name}")
    config = load_config()
    X, y = load_and_preprocess_data(split=False)

    # Set output path for this strategy
    output_path = f"results/history_{strategy_name}.json"
    config["output_path"] = output_path

    print(f"🤖 Running Active Learning with strategy: {strategy_name}")
    history_dict = active_learning_loop(X, y, config, strategy=strategy_name)

    # Save result (although already saved inside loop)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(history_dict, f)

    print(f"✅ Done. Saved to {output_path}\n")


def main():
    strategies = ["entropy", "random", "least_confidence","margin_sampling"]  # Add more if needed
    for strategy in strategies:
        run_and_save(strategy)


if __name__ == "__main__":
    main()
