# src/models.py

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct
import yaml


def load_model_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["model"]


def build_gp_model(config_path="config.yaml"):
    config = load_model_config(config_path)

    kernel_type = config.get("kernel", "RBF")
    length_scale = config.get("length_scale", 1.0)
    alpha = config.get("alpha", 1e-5)

    # Choose kernel
    if kernel_type == "RBF":
        kernel = RBF(length_scale=length_scale)
    elif kernel_type == "Matern":
        kernel = Matern(length_scale=length_scale, nu=1.5)
    elif kernel_type == "RationalQuadratic":
        kernel = RationalQuadratic(length_scale=length_scale, alpha=alpha)
    elif kernel_type == "DotProduct":
        kernel = DotProduct()
    else:
        raise ValueError(f"Unsupported kernel: {kernel_type}")

    return GaussianProcessClassifier(
        kernel=kernel,
        random_state=42,
        max_iter_predict=100
    )
