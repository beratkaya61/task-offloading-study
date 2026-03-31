def calculate_reward(action, delay, energy, task, device, local_energy_pred):
    """
    Calculates the reward (or penalty) for an offloading decision made by the RL Agent.
    Calibrated based on Phase 5 findings to reduce over-reliance on semantic shaping.
    """
    base_reward = 100.0
    reward = base_reward
    
    # 1. Core Objectives: Minimize Delay and Energy (Strengthened based on Phase 5)
    # Increased weights to ensure physical reality isn't ignored without shaping
    reward -= (delay * 35.0) 
    reward -= (energy * 5.0)
    
    # 2. LLM Semantic Alignment Bonus (With Confidence Thresholding)
    semantic = task.semantic_analysis
    llm_rec = semantic.get('recommended_target', 'edge') if semantic else 'edge'
    llm_confidence = semantic.get('confidence', 0.5) if semantic else 0.5
    
    # Apply thresholding: if LLM is not sure (< 0.7), lower the bonus impact
    conf_factor = llm_confidence if llm_confidence > 0.7 else llm_confidence * 0.4
    
    if llm_rec == 'local' and action == 0:
        reward += 20.0 * conf_factor
    elif llm_rec == 'edge' and 1 <= action <= 4:
        reward += 15.0 * conf_factor
    elif llm_rec == 'cloud' and action == 5:
        reward += 15.0 * conf_factor
    else:
        reward -= 12.0 * conf_factor # Slightly higher penalty for disobeying high-conf rec
        
    # 3. Penalize Cloud Cost
    if action == 5:
        reward -= 20.0 # Adjusted cost
        
    priority_score = semantic.get('priority_score', 0.5) if semantic else 0.5
    
    # 4. Deadline Miss Penalty (Normalized by priority)
    if delay > task.deadline:
        reward -= 60.0 * priority_score # Miss is worse for critical tasks
        
    # 5. Battery Awareness (Exponential Penalty - Calibrated Phase 5)
    battery_pct = (device.battery / 10000.0) * 100.0 if hasattr(device, 'battery') else 100.0
    
    if battery_pct < 25.0:
        # Exponential growth for penalty as battery approaches zero
        severity = (25.0 - battery_pct) / 25.0
        if action != 0: # Penalize offloading when battery is critical
            reward -= 40.0 * (severity ** 2)
        else:
            reward += 10.0 * severity # Reward local conservation
            
    # 6. Granular Partial Offloading Utilities (Awareness of splits)
    if 1 <= action <= 3:
        local_delay_only = task.cpu_cycles / 1e9
        if delay < local_delay_only:
            reward += 12.0 * ((local_delay_only - delay) / local_delay_only)
        
        # Energy saving utility
        reward += 5.0 * (1.0 - energy / max(1e-5, local_energy_pred))
        
    return reward

