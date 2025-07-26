# scripts/run_baseline.py

import yaml
from sklearn.metrics import accuracy_score, f1_score, classification_report
from preprocessing import load_and_preprocess_data
from models import build_gp_model

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    print("🔧 Loading configuration...")
    config = load_config()

    print("📊 Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    print("🧠 Building Gaussian Process model...")
    model = build_gp_model()

    print("🚀 Training...")
    model.fit(X_train, y_train)

    print("📈 Evaluating...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\n✅ Accuracy: {acc:.4f}")
    print(f"✅ Macro F1 Score: {f1:.4f}")
    print("\nDetailed Report:\n", classification_report(y_test, y_pred))

    # Optional: Save metrics to file
    if "evaluation" in config:
        output_path = config["evaluation"].get("save_results_to", None)
        if output_path:
            import json
            results = {"accuracy": acc, "macro_f1": f1}
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"📁 Results saved to {output_path}")

if __name__ == "__main__":
    main()
