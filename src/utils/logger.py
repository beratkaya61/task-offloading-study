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
        # Save to JSON
        json_path = os.path.join(self.log_dir, f"{prefix}_{self.run_id[:8]}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.log_data, f, indent=4)
        
        # Flatten dictionary for CSV
        flat_data = {"run_id": self.run_id, "timestamp": self.timestamp}
        if "config" in self.log_data:
            for k, v in self.log_data["config"].items():
                flat_data[f"config_{k}"] = v
        if "metrics" in self.log_data:
            for k, v in self.log_data["metrics"].items():
                flat_data[f"metric_{k}"] = v
                
        # Save to CSV
        csv_path = os.path.join(self.log_dir, "master_experiments.csv")
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_data.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(flat_data)
        
        return json_path, csv_path
