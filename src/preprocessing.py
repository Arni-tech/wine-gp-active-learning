# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml


def load_config(config_path="config.yaml"):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def bin_quality(value, bin_config):
    """Map wine quality scores to class labels based on bins in config."""
    for bin_def in bin_config:
        if bin_def["min"] <= value <= bin_def["max"]:
            return bin_def["label"]
    raise ValueError(f"Value {value} does not fit into any bin range.")


def apply_resampling(X, y, method="smote"):
    """Apply resampling to balance classes."""
    if method == "smote":
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        return smote.fit_resample(X, y)
    else:
        raise NotImplementedError(f"Resampling method '{method}' is not supported.")


def load_and_preprocess_data(config_path="config.yaml", split=True):
    """
    Load, preprocess, and optionally split the dataset.
    
    Parameters:
        config_path (str): Path to YAML config.
        split (bool): Whether to split into train/test.
    
    Returns:
        If split: X_train, X_test, y_train, y_test
        Else: X, y
    """
    config = load_config(config_path)
    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]

    # Load dataset
    df = pd.read_csv(data_cfg["path"])

    # Apply quality binning
    bin_config = prep_cfg["bin_quality"]
    df["quality_label"] = df["quality"].apply(lambda x: bin_quality(x, bin_config))

    # Extract features and labels
    X = df.drop(columns=["quality", "quality_label"])
    y = df["quality_label"]

    # Normalize features
    if prep_cfg.get("normalize", False):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = X.values

    # Apply train/test split or return full set
    if split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=data_cfg.get("test_size", 0.2),
            random_state=data_cfg.get("random_seed", 42),
            stratify=y
        )

        # Resample training set if enabled
        if prep_cfg.get("resample", {}).get("enabled", False):
            method = prep_cfg["resample"].get("method", "smote")
            X_train, y_train = apply_resampling(X_train, y_train, method=method)

        return X_train, X_test, y_train, y_test

    else:
        # Apply resampling on full set if enabled
        if prep_cfg.get("resample", {}).get("enabled", False):
            method = prep_cfg["resample"].get("method", "smote")
            X, y = apply_resampling(X, y, method=method)

        return X, y
