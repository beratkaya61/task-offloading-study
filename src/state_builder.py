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
    
    # 3. Handle LLM One-Hot Feature
    llm_rec = 'edge'
    if task.semantic_analysis and 'recommended_target' in task.semantic_analysis:
        llm_rec = task.semantic_analysis['recommended_target']
        
    if llm_rec == 'local':
        llm_onehot = [1.0, 0.0, 0.0]
    elif llm_rec == 'cloud':
        llm_onehot = [0.0, 0.0, 1.0]
    else:
        llm_onehot = [0.0, 1.0, 0.0]
        
    return np.array([snr_norm, size_norm, cpu_norm, batt_norm, load_norm] + llm_onehot, dtype=np.float32)
