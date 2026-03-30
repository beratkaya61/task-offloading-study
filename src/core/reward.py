def calculate_reward(action, delay, energy, task, device, local_energy_pred):
    """
    Calculates the reward (or penalty) for an offloading decision made by the RL Agent.
    Extracted from the RL Environment to be universally testable and tunable.
    """
    base_reward = 100.0
    reward = base_reward
    
    # 1. Core Objectives: Minimize Delay and Energy
    reward -= (delay * 20.0)
    reward -= (energy * 2.0)
    
    # 2. LLM Semantic Alignment Bonus
    semantic = task.semantic_analysis
    llm_rec = semantic.get('recommended_target', 'edge') if semantic else 'edge'
    llm_confidence = semantic.get('confidence', 0.5) if semantic else 0.5
    
    if llm_rec == 'local' and action == 0:
        reward += 20.0 * llm_confidence
    elif llm_rec == 'edge' and 1 <= action <= 4:
        reward += 15.0 * llm_confidence
    elif llm_rec == 'cloud' and action == 5:
        reward += 15.0 * llm_confidence
    else:
        reward -= 10.0 * llm_confidence # Penalty for disobeying LLM strongly
        
    # 3. Penalize Cloud Cost
    if action == 5:
        reward -= 25.0
        
    priority_score = semantic.get('priority_score', 0.5) if semantic else 0.5
    
    # 4. Deadline Miss Penalty
    if delay > task.deadline:
        reward -= 50.0 * priority_score # Miss is worse for critical tasks
        
    # 5. Battery Awareness Check
    battery_pct = (device.battery / 10000.0) * 100.0 if hasattr(device, 'battery') else 100.0
    
    if battery_pct < 30.0:
        if action == 0:
            reward += 15.0 * (1.0 - battery_pct/100.0) # Bonus for local conservation
        elif 1 <= action <= 3:
            reward += 10.0 * (1.0 - battery_pct/100.0) # Bonus for partial conservation
        else:
            reward -= 30.0 * (1.0 - battery_pct/100.0) # Penalty for draining / leaving edge overloaded
            
    # 6. Granular Partial Offloading Utilities
    if 1 <= action <= 3:
        local_delay_only = task.cpu_cycles / 1e9
        if delay < local_delay_only:
            # Reward for speeding up the computation over pure local
            reward += 10.0 * ((local_delay_only - delay) / local_delay_only) + 3.0
        
        # Reward for saving overall energy compared to pure local computation
        reward += 2.0 * (1.0 - energy / max(1e-5, local_energy_pred * 0.5))
        
    # 7. Local Processing Efficiency Bonus
    if action == 0 and energy < 0.1:
        reward += 8.0
        if battery_pct > 40.0:
            reward += 5.0
            
    return reward
