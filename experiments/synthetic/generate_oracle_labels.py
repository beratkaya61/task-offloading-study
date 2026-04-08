from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.pretrain_policy import generate_oracle_dataset


if __name__ == "__main__":
    result = generate_oracle_dataset("configs/synthetic/oracle_labeling.yaml")
    print(f"[INFO] Oracle dataset CSV: {result['csv_path']}")
    print(f"[INFO] Oracle summary report: {result['report_path']}")
    print(f"[INFO] Total rows: {result['num_rows']}")
