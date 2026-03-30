import numpy as np

def generate_action_prior(semantic_analysis):
    """
    Converts semantic analysis (recommendation and confidence) into a 6D probability distribution over actions.
    Indices: [Local, Edge_25, Edge_50, Edge_75, Edge_100, Cloud]
    This replaces the naive one-hot encoding, providing the RL agent with a 'Semantic Prior' distribution.
    """
    # Default uniform distribution if no analysis
    if not semantic_analysis:
        return np.ones(6) / 6.0
        
    target = semantic_analysis.get('recommended_target', 'edge')
    conf = semantic_analysis.get('confidence', 0.5)
    
    # Extract structural constraints safely
    conf = max(0.0, min(1.0, conf)) # clamp 0-1
    
    # Base smoothing (1 - conf) distributed among all actions
    prior = np.ones(6) * ((1.0 - conf) / 6.0)
    
    if target == 'local':
        # Boost Local (Index 0)
        prior[0] += conf
    elif target == 'cloud':
        # Boost Cloud (Index 5)
        prior[5] += conf
    else: # target == 'edge'
        # Boost Edge Actions (Indices 1 to 4)
        # Distribute the confidence heavily towards edge processing with a slight bias towards Full Edge
        prior[1] += conf * 0.15 # 25% Edge
        prior[2] += conf * 0.25 # 50% Edge
        prior[3] += conf * 0.30 # 75% Edge
        prior[4] += conf * 0.30 # Full Edge
        
    # Ensure normalization against floating point drift
    prior = prior / np.sum(prior)
    return prior.astype(np.float32)

def log_semantic_explanation(task, action, prior):
    """
    Creates a detailed explanation log of the task, LLM prior, and chosen action.
    This builds an Experience Bank for Phase 10 reflection or analysis.
    """
    import json
    import os
    from datetime import datetime
    
    log_dir = "phase_reports/semantic_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    semantic = task.semantic_analysis if task.semantic_analysis else {}
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "task_id": getattr(task, 'id', -1),
        "type": task.task_type.name if hasattr(task, 'task_type') else "UNKNOWN",
        "llm_recommendation": semantic.get('recommended_target', 'none'),
        "llm_confidence": semantic.get('confidence', 0.0),
        "llm_reason": semantic.get('reason', ''),
        "produced_prior": [round(float(p), 4) for p in prior],
        "final_action_taken": int(action)
    }
    
    log_path = os.path.join(log_dir, "explanation_bank.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
