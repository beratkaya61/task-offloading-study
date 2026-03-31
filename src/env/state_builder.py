import numpy as np
import math

def build_state(device, task, edge_servers, channel):
    """
    Builds the normalized state vector for the RL agent.
    Features: [SNR, Task Size, CPU Cycles, Battery %, Edge Load, LLM-Local, LLM-Edge, LLM-Cloud]
    Uses common logic for both the physical simulation and the training environment.
    """
    if not device or not task:
        return np.zeros((8,), dtype=np.float32)

    # 1. Calculate Data Rate
    if edge_servers:
        closest_edge = min(edge_servers, key=lambda e: math.dist(device.location, e.location))
        datarate, _ = channel.calculate_datarate(device, closest_edge)
    else:
        closest_edge = None
        datarate = 10e6 # Fallback datarate
        
    # 2. Normalize Continuous Features
    snr_norm = min(1.0, datarate / 50e6) # Normalize by 50 Mbps max
    size_norm = min(1.0, task.size_bits / 10e6) # Max 10MB
    cpu_norm = min(1.0, task.cpu_cycles / 1e10) # Max 10B cycles
    
    battery_capacity = getattr(device, 'battery', 10000.0)
    batt_norm = min(1.0, max(0.0, battery_capacity / 10000.0))
    
    load_norm = min(1.0, closest_edge.current_load / 10.0) if closest_edge else 0.0
    
    # 3. Handle LLM Semantic Prior (6D vector)
    from src.agents.semantic_prior import generate_action_prior
    prior_vector = generate_action_prior(task.semantic_analysis)
    
    return np.concatenate((
        np.array([snr_norm, size_norm, cpu_norm, batt_norm, load_norm], dtype=np.float32), 
        prior_vector
    )).astype(np.float32)
