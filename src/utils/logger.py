import json
import csv
import uuid
import datetime
import os

class ExperimentLogger:
    def __init__(self, log_dir="results/raw"):
        """
        Initializes the experiment logger, generating a unique run ID and timestamp.
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.run_id = str(uuid.uuid4())
        self.timestamp = datetime.datetime.now().isoformat()
        self.log_data = {
            "run_id": self.run_id,
            "timestamp": self.timestamp
        }
    
    def log_config(self, config: dict):
        """Logs the experiment configuration."""
        self.log_data["config"] = config
    
    def log_metrics(self, metrics: dict):
        """Logs the experiment metrics (e.g., avg_latency, p95_latency, success_rate)."""
        self.log_data["metrics"] = metrics
        
    def save(self, prefix="run"):
        """
        Saves the logged data to a JSON file and appends it to a master CSV file.
        Returns the paths of the saved JSON and CSV files.
        """
        import hashlib
        
        # 1. Enforce Config Hash & Requirements
        if "config" not in self.log_data:
            self.log_data["config"] = {}
            
        if "config_hash" not in self.log_data["config"]:
            dict_str = json.dumps(self.log_data["config"], sort_keys=True)
            self.log_data["config"]["config_hash"] = hashlib.md5(dict_str.encode()).hexdigest()
            
        # 2. Save JSON
        json_path = os.path.join(self.log_dir, f"{prefix}_{self.run_id[:8]}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.log_data, f, indent=4)
        
        # 3. CSV Flattening & Schema Enforcement (Phase 1 reqs)
        required_config_keys = ["config_hash", "seed", "model_type", "semantic_mode", "total_tasks"]
        required_metric_keys = ["success_rate", "avg_latency", "p95_latency", "avg_energy", "fairness", "jitter", "qoe", "decision_overhead"]
        
        flat_data = {"run_id": self.run_id, "timestamp": self.timestamp}
        
        for k in required_config_keys:
            flat_data[f"config_{k}"] = self.log_data["config"].get(k, "None")
        for k, v in self.log_data["config"].items():
            if f"config_{k}" not in flat_data: flat_data[f"config_{k}"] = v
            
        if "metrics" not in self.log_data:
            self.log_data["metrics"] = {}
            
        for k in required_metric_keys:
            flat_data[f"metric_{k}"] = self.log_data["metrics"].get(k, "None")
        for k, v in self.log_data["metrics"].items():
            if f"metric_{k}" not in flat_data: flat_data[f"metric_{k}"] = v
                
        # 4. Save to Master CSV smartly (Handling Header differences)
        csv_path = os.path.join(self.log_dir, "master_experiments.csv")
        file_exists = os.path.isfile(csv_path)
        
        fieldnames = list(flat_data.keys())
        
        # To avoid schema drifting failures, we will read the old header if it exists
        if file_exists:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    # Dynamically expand header for any new keys to preserve old CSV
                    for col in fieldnames:
                        if col not in header:
                            header.append(col)
                    fieldnames = header

        # Rewrite row to match extended Fieldnames safely
        safe_data = {k: flat_data.get(k, "") for k in fieldnames}
        
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(safe_data)
        
        return json_path, csv_path
