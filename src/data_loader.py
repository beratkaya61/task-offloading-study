import random
import pandas as pd
import numpy as np
import os

class DataLoader:
    """
    Data Loader for Google Cluster Trace and Didi Gaia datasets.
    Supports both real CSV files and mock data generation.
    """
    
    @staticmethod
    def load_google_cluster_trace(filepath=None, num_tasks=1000):
        """
        Loads task attributes from Google Cluster Trace CSV.
        If filepath is None or file doesn't exist, generates mock data.
        
        Expected CSV format:
        timestamp,task_id,cpu_request,memory_request,task_type
        
        Returns: pandas DataFrame with columns:
        - task_id: Unique task identifier
        - submit_time: Task arrival time (seconds)
        - cpu_request: CPU cycles required
        - ram_request: RAM in MB
        - task_type: Task classification
        """
        
        if filepath and os.path.exists(filepath):
            print(f"Loading REAL Google Cluster Trace from: {filepath}")
            try:
                df = pd.read_csv(filepath)
                
                # Normalize to expected format
                if 'timestamp' in df.columns:
                    df['submit_time'] = (df['timestamp'] - df['timestamp'].min())  # Normalize to 0
                
                # Convert CPU request to cycles
                if 'cpu_request' in df.columns:
                    df['cpu_request'] = df['cpu_request'] * 1e9  # Convert to cycles
                
                print(f"Loaded {len(df)} real tasks from CSV")
                return df
                
            except Exception as e:
                print(f"Error loading CSV: {e}. Falling back to mock data.")
        
        # Fallback to Mock Data
        print(f"Generating MOCK Google Cluster Trace ({num_tasks} tasks)...")
        
        data = {
            'task_id': range(num_tasks),
            'submit_time': np.sort(np.random.exponential(scale=10.0, size=num_tasks)),
            'cpu_request': np.random.pareto(a=2.0, size=num_tasks) * 1e9,  # Pareto for realistic distribution
            'ram_request': np.random.uniform(128, 4096, size=num_tasks),
            'task_type': np.random.choice(
                ['AI_INFERENCE', 'VIDEO_TRANSCODE', 'IOT_SENSING', 'CRITICAL_HEALTH'],
                size=num_tasks,
                p=[0.2, 0.3, 0.4, 0.1]
            )
        }
        
        return pd.DataFrame(data)

    @staticmethod
    def load_didi_gaia_mobility(filepath=None, num_users=20, duration=1000):
        """
        Loads mobility traces from Didi Gaia CSV.
        If filepath is None or file doesn't exist, generates mock mobility.
        
        Expected CSV format:
        user_id,timestamp,latitude,longitude
        
        Returns: Dictionary mapping UserID -> List of (x,y) coordinates
        """
        
        if filepath and os.path.exists(filepath):
            print(f"Loading REAL Didi Gaia Mobility from: {filepath}")
            try:
                df = pd.read_csv(filepath)
                
                # Group by user_id
                mobility_traces = {}
                for user_id in df['user_id'].unique()[:num_users]:
                    user_data = df[df['user_id'] == user_id].sort_values('timestamp')
                    
                    # Normalize coordinates to 0-1000 range (simulation space)
                    lat_min, lat_max = user_data['latitude'].min(), user_data['latitude'].max()
                    lon_min, lon_max = user_data['longitude'].min(), user_data['longitude'].max()
                    
                    x = ((user_data['latitude'] - lat_min) / (lat_max - lat_min) * 1000).tolist()
                    y = ((user_data['longitude'] - lon_min) / (lon_max - lon_min) * 1000).tolist()
                    
                    mobility_traces[user_id] = list(zip(x, y))
                
                print(f"Loaded mobility for {len(mobility_traces)} users")
                return mobility_traces
                
            except Exception as e:
                print(f"Error loading CSV: {e}. Falling back to mock data.")
        
        # Fallback to Mock Data
        print(f"Generating MOCK Didi Gaia Mobility ({num_users} users)...")
        
        mobility_traces = {}
        
        for user_id in range(num_users):
            # Start at random location
            x, y = random.uniform(0, 1000), random.uniform(0, 1000)
            path = []
            
            # Simulate car movement with realistic turns
            velocity_x = random.uniform(-10, 10)
            velocity_y = random.uniform(-10, 10)
            
            for t in range(duration):
                # Update position
                x += velocity_x
                y += velocity_y
                
                # Boundary reflection
                if x < 0 or x > 1000: velocity_x *= -1
                if y < 0 or y > 1000: velocity_y *= -1
                
                # Random direction change (5% probability)
                if random.random() < 0.05:
                    velocity_x = random.uniform(-10, 10)
                    velocity_y = random.uniform(-10, 10)
                
                path.append((x, y))
            
            mobility_traces[user_id] = path
            
        return mobility_traces
    
    @staticmethod
    def save_sample_csv_format():
        """
        Creates sample CSV files showing the expected format.
        Useful for users who want to use real data.
        """
        # Sample Google Trace
        sample_tasks = pd.DataFrame({
            'timestamp': [0, 10, 20, 30],
            'task_id': [1, 2, 3, 4],
            'cpu_request': [0.5, 1.0, 0.3, 2.0],  # In billions of cycles
            'memory_request': [512, 1024, 256, 2048],
            'task_type': ['AI_INFERENCE', 'VIDEO_TRANSCODE', 'IOT_SENSING', 'CRITICAL_HEALTH']
        })
        sample_tasks.to_csv('sample_google_trace.csv', index=False)
        
        # Sample Mobility
        sample_mobility = pd.DataFrame({
            'user_id': [0, 0, 0, 1, 1, 1],
            'timestamp': [0, 1, 2, 0, 1, 2],
            'latitude': [39.9042, 39.9043, 39.9044, 40.7128, 40.7129, 40.7130],
            'longitude': [116.4074, 116.4075, 116.4076, -74.0060, -74.0061, -74.0062]
        })
        sample_mobility.to_csv('sample_didi_mobility.csv', index=False)
        
        print("Sample CSV files created: sample_google_trace.csv, sample_didi_mobility.csv")

if __name__ == "__main__":
    # Test the loader
    print("=== Testing Data Loader ===\n")
    
    # Test Task Data
    tasks = DataLoader.load_google_cluster_trace(num_tasks=10)
    print(tasks.head())
    print(f"\nTask Types: {tasks['task_type'].unique()}\n")
    
    # Test Mobility Data
    mobility = DataLoader.load_didi_gaia_mobility(num_users=2, duration=5)
    print(f"User 0 Path (first 5 steps): {mobility[0][:5]}\n")
    
    # Create sample CSVs
    DataLoader.save_sample_csv_format()
