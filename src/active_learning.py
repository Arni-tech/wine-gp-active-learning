# src/active_learning.py

import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
from models import build_gp_model  # ✅ corrected relative import


def compute_uncertainty(proba_matrix):
    """
    Compute entropy-based uncertainty.
    Shape of proba_matrix: (n_samples, n_classes)
    """
    return entropy(proba_matrix.T)  # Transpose for scipy format (classes, samples)


def active_learning_loop(X, y, config, strategy="entropy"):
    """
    Active learning loop with entropy or random sampling strategy.

    Args:
        X (np.ndarray): Features
        y (np.ndarray): Labels
        config (dict): Configuration loaded from config.yaml
        strategy (str): Sampling strategy ("entropy" or "random")

    Returns:
        dict: Dictionary containing strategy and learning curve history
    """

    # Config parameters
    seed_size = config["active_learning"]["initial_label_size"]
    query_size = config["active_learning"]["query_batch_size"]
    max_queries = config["active_learning"]["max_queries"]
    test_size = config["data"].get("test_size", 0.2)
    random_state = config["data"].get("random_seed", 42)

    # Shuffle and split fixed test set
    X, y = shuffle(X, y, random_state=random_state)
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    y_pool = y_pool.to_numpy()
    y_test = y_test.to_numpy()

    # Initialize labeled/unlabeled pools
    labeled_idx = np.random.choice(len(X_pool), size=seed_size, replace=False)
    unlabeled_idx = np.setdiff1d(np.arange(len(X_pool)), labeled_idx)

    history = []

    for i in range(0, max_queries, query_size):
        model = build_gp_model()
        model.fit(X_pool[labeled_idx], y_pool[labeled_idx])

        # Evaluate on fixed test set
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        print(f"[{strategy.title()} Iteration {i//query_size + 1}] Accuracy: {acc:.4f}, F1: {f1:.4f}")
        history.append((int(len(labeled_idx)), float(acc), float(f1)))

        if len(unlabeled_idx) == 0:
            break

        # Select new samples
        if strategy == "entropy":
            proba = model.predict_proba(X_pool[unlabeled_idx])
            uncertainty = compute_uncertainty(proba)
            query_indices = np.argsort(uncertainty)[-query_size:]

        elif strategy == "least_confidence":
            proba = model.predict_proba(X_pool[unlabeled_idx])
            confidence = np.max(proba, axis=1)
            uncertainty = 1 - confidence
            query_indices = np.argsort(uncertainty)[-query_size:]

        elif strategy == "random":
            query_indices = np.random.choice(len(unlabeled_idx), size=query_size, replace=False)
        elif strategy == "margin_sampling":
            proba = model.predict_proba(X_pool[unlabeled_idx])
            sorted_probs = np.sort(proba, axis=1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            query_indices = np.argsort(margin)[:query_size]

        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Must be one of 'entropy', 'least_confidence', 'random'.")

        selected_idx = unlabeled_idx[query_indices]
        labeled_idx = np.concatenate([labeled_idx, selected_idx])
        unlabeled_idx = np.setdiff1d(unlabeled_idx, selected_idx)

    # Bundle results
    history_dict = {
        "strategy": strategy,
        "results": history
    }

    # Save output file
    output_path = config.get("output_path", f"results/history_{strategy}.json")
    with open(output_path, "w") as f:
        json.dump(history_dict, f)

    return history_dict
