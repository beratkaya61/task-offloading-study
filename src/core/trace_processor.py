"""
Trace Processor for Faz 6: Trace-driven Training

Loads real mobility traces (Didi Gaia dataset) and preprocesses them
into training episodes for PPO agent.

Key responsibilities:
- Parse trace data (mobility patterns, device locations)
- Generate training tasks based on real trace characteristics
- Balance dataset (priority distribution, task complexity)
- Create validation/test splits
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


@dataclass
class TraceTask:
    """Single task from a trace"""
    task_id: int
    device_id: int
    arrival_time: float
    deadline: float
    data_size: int  # KB
    cpu_cycles: int
    priority: int  # 0-3 (0=low, 3=high)
    location: Tuple[float, float]  # (x, y) coordinates
    
    def to_dict(self):
        return {
            'task_id': self.task_id,
            'device_id': self.device_id,
            'arrival_time': self.arrival_time,
            'deadline': self.deadline,
            'data_size': self.data_size,
            'cpu_cycles': self.cpu_cycles,
            'priority': self.priority,
            'location': self.location
        }


@dataclass
class TraceEpisode:
    """Single training episode from traces"""
    episode_id: int
    tasks: List[TraceTask]
    trace_name: str
    device_density: int  # number of active devices
    
    def __len__(self):
        return len(self.tasks)


class TraceProcessor:
    """Main trace processor for Faz 6"""
    
    def __init__(self, trace_dir: Optional[str] = None, seed: int = 42):
        """
        Args:
            trace_dir: Directory containing trace files (CSV format)
            seed: Random seed for reproducibility
        """
        self.trace_dir = Path(trace_dir) if trace_dir else Path('data/traces')
        self.seed = seed
        np.random.seed(seed)
        self.episodes = []
        self.metadata = {}
        
    def load_traces(self, pattern: str = "*.csv") -> List[pd.DataFrame]:
        """
        Load trace files matching pattern.
        
        Args:
            pattern: Glob pattern for trace files
            
        Returns:
            List of DataFrames from trace files
        """
        if not self.trace_dir.exists():
            print(f"⚠️ Trace directory not found: {self.trace_dir}")
            print("   → Using synthetic traces instead")
            return self._generate_synthetic_traces()
        
        traces = []
        for trace_file in self.trace_dir.glob(pattern):
            try:
                df = pd.read_csv(trace_file)
                traces.append(df)
                print(f"✅ Loaded trace: {trace_file.name} ({len(df)} records)")
            except Exception as e:
                print(f"❌ Error loading {trace_file}: {e}")
        
        return traces if traces else self._generate_synthetic_traces()
    
    def _generate_synthetic_traces(self, n_devices: int = 20, 
                                   n_tasks: int = 500) -> List[pd.DataFrame]:
        """
        Generate synthetic traces mimicking Didi Gaia characteristics.
        
        Characteristics:
        - Mobile devices with varying speeds
        - Task arrivals follow Poisson process
        - Task priorities: 0-3
        - Data sizes: 100KB to 10MB
        - Deadlines: 500ms to 5s
        """
        print(f"🔄 Generating synthetic traces: {n_devices} devices, ~{n_tasks} tasks")
        
        np.random.seed(self.seed)
        
        # Simulate device movements (simplified Didi-like patterns)
        data = []
        for task_id in range(n_tasks):
            device_id = np.random.randint(0, n_devices)
            
            # Task characteristics
            arrival_time = task_id * np.random.exponential(0.5)  # Poisson-like arrivals
            deadline = arrival_time + np.random.uniform(0.5, 5.0)  # 500ms - 5s
            data_size = int(np.random.exponential(500) + 100)  # 100KB - 10MB (exponential)
            cpu_cycles = int(np.random.exponential(5e8) + 1e8)  # 100M - 1B cycles
            priority = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])
            
            # Location (simplified grid: 0-100 in both dimensions)
            x = (device_id % 5) * 20 + np.random.normal(0, 5)
            y = (device_id // 5) * 20 + np.random.normal(0, 5)
            x = np.clip(x, 0, 100)
            y = np.clip(y, 0, 100)
            
            data.append({
                'task_id': task_id,
                'device_id': device_id,
                'arrival_time': arrival_time,
                'deadline': deadline,
                'data_size': data_size,
                'cpu_cycles': cpu_cycles,
                'priority': priority,
                'location_x': x,
                'location_y': y
            })
        
        df = pd.DataFrame(data)
        return [df]
    
    def preprocess_traces(self, traces: List[pd.DataFrame], 
                         normalize: bool = True) -> List[pd.DataFrame]:
        """
        Preprocess traces (normalize, filter outliers, etc.)
        
        Args:
            traces: List of raw trace DataFrames
            normalize: Whether to normalize features
            
        Returns:
            List of preprocessed DataFrames
        """
        processed = []
        
        for trace_df in traces:
            df = trace_df.copy()
            
            # Filter unrealistic values
            df = df[df['data_size'] > 0]
            df = df[df['cpu_cycles'] > 0]
            df = df[df['deadline'] > df['arrival_time']]
            df = df[df['priority'].isin([0, 1, 2, 3])]
            
            # Normalize features (optional)
            if normalize:
                df['data_size_norm'] = (df['data_size'] - df['data_size'].mean()) / df['data_size'].std()
                df['cpu_cycles_norm'] = (df['cpu_cycles'] - df['cpu_cycles'].mean()) / df['cpu_cycles'].std()
                df['deadline_norm'] = (df['deadline'] - df['deadline'].mean()) / df['deadline'].std()
            
            processed.append(df)
            print(f"   Preprocessed: {len(df)} tasks (removed {len(trace_df)-len(df)} outliers)")
        
        return processed
    
    def generate_episodes(self, traces: List[pd.DataFrame], 
                         tasks_per_episode: int = 50,
                         n_episodes: int = 100) -> List[TraceEpisode]:
        """
        Generate training episodes from preprocessed traces.
        
        Args:
            traces: List of preprocessed DataFrames
            tasks_per_episode: Tasks per training episode
            n_episodes: Number of episodes to generate
            
        Returns:
            List of TraceEpisode objects
        """
        episodes = []
        
        # Combine all traces
        all_tasks = pd.concat(traces, ignore_index=True).reset_index(drop=True)
        
        print(f"🔄 Generating {n_episodes} episodes ({tasks_per_episode} tasks/episode)")
        print(f"   Total tasks available: {len(all_tasks)}")
        
        for ep_id in range(n_episodes):
            # Sample tasks for this episode
            if len(all_tasks) >= tasks_per_episode:
                sampled_tasks_df = all_tasks.sample(n=tasks_per_episode, replace=False)
            else:
                # If fewer tasks, resample with replacement
                sampled_tasks_df = all_tasks.sample(n=tasks_per_episode, replace=True)
            
            # Sort by arrival time to maintain temporal order
            sampled_tasks_df = sampled_tasks_df.sort_values('arrival_time').reset_index(drop=True)
            
            # Convert to TraceTask objects
            trace_tasks = []
            for idx, row in sampled_tasks_df.iterrows():
                task = TraceTask(
                    task_id=int(row['task_id']),
                    device_id=int(row['device_id']),
                    arrival_time=float(row['arrival_time']),
                    deadline=float(row['deadline']),
                    data_size=int(row['data_size']),
                    cpu_cycles=int(row['cpu_cycles']),
                    priority=int(row['priority']),
                    location=(float(row.get('location_x', 50)), 
                             float(row.get('location_y', 50)))
                )
                trace_tasks.append(task)
            
            # Create episode
            device_density = len(sampled_tasks_df['device_id'].unique())
            episode = TraceEpisode(
                episode_id=ep_id,
                tasks=trace_tasks,
                trace_name=f"synthetic_didi_ep{ep_id}",
                device_density=device_density
            )
            episodes.append(episode)
        
        print(f"✅ Generated {len(episodes)} training episodes")
        self.episodes = episodes
        return episodes
    
    def split_episodes(self, train_ratio: float = 0.8, 
                       val_ratio: float = 0.1) -> Tuple[List[TraceEpisode], 
                                                         List[TraceEpisode], 
                                                         List[TraceEpisode]]:
        """
        Split episodes into train/val/test sets.
        
        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation (rest goes to test)
            
        Returns:
            Tuple of (train_episodes, val_episodes, test_episodes)
        """
        n_episodes = len(self.episodes)
        n_train = int(n_episodes * train_ratio)
        n_val = int(n_episodes * val_ratio)
        
        indices = np.random.permutation(n_episodes)
        
        train_eps = [self.episodes[i] for i in indices[:n_train]]
        val_eps = [self.episodes[i] for i in indices[n_train:n_train+n_val]]
        test_eps = [self.episodes[i] for i in indices[n_train+n_val:]]
        
        print(f"📊 Episode split: Train={len(train_eps)}, Val={len(val_eps)}, Test={len(test_eps)}")
        
        return train_eps, val_eps, test_eps
    
    def save_episodes(self, episodes: List[TraceEpisode], 
                     output_path: str) -> None:
        """Save episodes to JSON for reproducibility"""
        data = {
            'episodes': [
                {
                    'episode_id': ep.episode_id,
                    'tasks': [task.to_dict() for task in ep.tasks],
                    'trace_name': ep.trace_name,
                    'device_density': ep.device_density
                }
                for ep in episodes
            ],
            'metadata': {
                'n_episodes': len(episodes),
                'seed': self.seed
            }
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saved {len(episodes)} episodes to {output_path}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about loaded traces"""
        if not self.episodes:
            return {}
        
        all_tasks = []
        for ep in self.episodes:
            all_tasks.extend(ep.tasks)
        
        data_sizes = [t.data_size for t in all_tasks]
        cpu_cycles = [t.cpu_cycles for t in all_tasks]
        priorities = [t.priority for t in all_tasks]
        deadlines = [t.deadline - t.arrival_time for t in all_tasks]
        
        return {
            'n_episodes': int(len(self.episodes)),
            'n_tasks_total': int(len(all_tasks)),
            'data_size': {
                'mean': float(np.mean(data_sizes)),
                'std': float(np.std(data_sizes)),
                'min': float(np.min(data_sizes)),
                'max': float(np.max(data_sizes))
            },
            'cpu_cycles': {
                'mean': float(np.mean(cpu_cycles)),
                'std': float(np.std(cpu_cycles)),
                'min': float(np.min(cpu_cycles)),
                'max': float(np.max(cpu_cycles))
            },
            'priority_distribution': {
                int(p): int(priorities.count(p)) for p in set(priorities)
            },
            'deadline': {
                'mean': float(np.mean(deadlines)),
                'std': float(np.std(deadlines)),
                'min': float(np.min(deadlines)),
                'max': float(np.max(deadlines))
            }
        }


if __name__ == "__main__":
    # Example usage
    processor = TraceProcessor(seed=42)
    
    # Load or generate traces
    traces = processor.load_traces()
    
    # Preprocess
    processed_traces = processor.preprocess_traces(traces)
    
    # Generate episodes
    episodes = processor.generate_episodes(processed_traces, 
                                           tasks_per_episode=50, 
                                           n_episodes=100)
    
    # Split
    train_eps, val_eps, test_eps = processor.split_episodes()
    
    # Save for training
    processor.save_episodes(train_eps, 'data/traces/train_episodes.json')
    processor.save_episodes(val_eps, 'data/traces/val_episodes.json')
    processor.save_episodes(test_eps, 'data/traces/test_episodes.json')
    
    # Statistics
    stats = processor.get_statistics()
    print("\n📊 Trace Statistics:")
    print(json.dumps(stats, indent=2))
