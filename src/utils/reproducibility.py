import os
import random
import numpy as np
import logging

try:
    import torch
except ImportError:
    torch = None

def set_seed(seed: int = 42, env=None):
    """
    Sets the random seed for reproducibility across core libraries:
    - python random
    - numpy
    - PyTorch
    - (Optional) Gymnasium environment
    """
    logging.info(f"Setting global seed to {seed} for reproducibility.")
    
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # OS Hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Torch
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # Environment
    if env is not None:
        try:
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        except Exception as e:
            logging.warning(f"Could not seed the environment: {e}")
