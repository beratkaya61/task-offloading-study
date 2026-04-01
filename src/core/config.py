import os

import yaml


def load_config(config_path="configs/synthetic_rl_training.yaml"):
    """
    Loads experiment configuration from a YAML file.
    """
    if not os.path.exists(config_path):
        print(f"[Config] Warning: {config_path} not found. Returning default config.")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    cfg = load_config()
    print("Loaded Config:", cfg)
