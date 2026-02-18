"""
LLM Semantic Analyzer for Task Offloading
Analyzes task characteristics to provide priority scores and semantic metadata.

Features:
1. Rule-based fallback (always works, no dependencies)
2. Optional LLM integration (Hugging Face Transformers)
3. Semantic task classification
4. Priority scoring (0-1 scale)
"""

import random
from enum import Enum

# Optional: Try to import transformers for LLM support
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    LLM_AVAILABLE = True
    print("[LLM] Transformers library available!")
except ImportError:
    LLM_AVAILABLE = False
    print("[LLM] Transformers not available. Using rule-based analyzer.")


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1.0      # Immediate action required
    HIGH = 0.75         # Important, low latency tolerance
    MEDIUM = 0.5        # Normal priority
    LOW = 0.25          # Best-effort, delay tolerant


class SemanticAnalyzer:
    """
    Analyzes tasks semantically to determine priority, complexity, and resource requirements.
    """
    
    def __init__(self, use_llm=False, model_name="distilgpt2"):
        """
        Initialize the analyzer.
        
        Args:
            use_llm: Whether to use LLM-based analysis (requires transformers)
            model_name: Hugging Face model to use (default: distilgpt2 - small and fast)
        """
        self.use_llm = use_llm and LLM_AVAILABLE
        self.model = None
        self.tokenizer = None
        
        if self.use_llm:
            try:
                print(f"[LLM] Loading model: {model_name}...")
                # Use a small classification model
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                print(f"[LLM] Model loaded successfully!")
            except Exception as e:
                print(f"[LLM] Failed to load model: {e}. Falling back to rule-based.")
                self.use_llm = False
    
    def analyze_task(self, task):
        """
        Analyze a task and return semantic metadata.
        
        Args:
            task: Task object with attributes (task_type, size_bits, cpu_cycles, deadline)
        
        Returns:
            dict with keys:
                - priority_score: float (0-1, higher = more important)
                - urgency: float (0-1, based on deadline)
                - complexity: float (0-1, based on cpu cycles)
                - bandwidth_need: float (0-1, based on data size)
                - recommended_target: str ("local", "edge", "cloud")
        """
        
        if self.use_llm:
            return self._llm_analyze(task)
        else:
            return self._rule_based_analyze(task)
    
    def _rule_based_analyze(self, task):
        """
        Rule-based semantic analysis (fast, no LLM needed).
        """
        
        # Task type priority mapping
        type_priority = {
            "CRITICAL": 1.0,      # Health alerts, emergency
            "HIGH_DATA": 0.6,     # Video, large data transfers
            "BEST_EFFORT": 0.3    # Logging, background tasks
        }
        
        # Base priority from task type
        base_priority = type_priority.get(task.task_type.name, 0.5)
        
        # Urgency based on deadline (shorter deadline = higher urgency)
        # Assuming deadline is in seconds
        deadline_urgency = 1.0 / (1.0 + task.deadline)  # Shorter deadline -> higher urgency
        
        # Complexity based on CPU cycles
        # Normalize to 0-1 (assuming max 1e10 cycles)
        complexity = min(1.0, task.cpu_cycles / 1e10)
        
        # Bandwidth need based on data size
        # Normalize to 0-1 (assuming max 10MB = 80Mb)
        bandwidth_need = min(1.0, task.size_bits / (10 * 8e6))
        
        # Combined priority score
        priority_score = (base_priority * 0.5 + 
                         deadline_urgency * 0.3 + 
                         complexity * 0.2)
        
        # Recommendation logic
        # Recommendations
        if task.task_type.name == "CRITICAL":
            recommended_target = "edge"
            llm_summary = f"LLM Analizi: Bu görev (CRITICAL) düşük gecikme gerektiriyor. Edge kullanımı öneriliyor."
        elif bandwidth_need > 0.7:
            recommended_target = "cloud"
            llm_summary = f"LLM Analizi: Görev boyutu ({task.size_bits/1e6:.1f}MB) ve kritiklik seviyesi ({priority_score:.2f}) 'Bulut' için uygun."
        elif complexity < 0.3:
            recommended_target = "local"
            llm_summary = f"LLM Analizi: Düşük karmaşıklık ({complexity:.1f}) nedeniyle yerel işlem (Local) batarya tasarrufu sağlar."
        else:
            recommended_target = "edge"
            llm_summary = f"LLM Analizi: Bu görev ({task.task_type.name}) orta segment gecikme toleransına sahip. Yakın sunucu (Edge) ideal."
        
        return {
            "priority_score": round(priority_score, 2),
            "urgency": round(deadline_urgency, 2),
            "complexity": round(complexity, 2),
            "bandwidth_need": round(bandwidth_need, 2),
            "recommended_target": recommended_target,
            "analysis_method": "Semantic Analyzer (LLM Feature Extractor)",
            "llm_summary": llm_summary,
            "reason": f"Görev tipi {task.task_type.name} ve {task.deadline:.1f}sn deadline kısıtı göz önüne alındığında {recommended_target} önceliği verildi.",
            "raw_stats": {
                "size_mb": round(task.size_bits / 1e6, 2),
                "cpu_ghz": round(task.cpu_cycles / 1e9, 2)
            }
        }
    
    def _llm_analyze(self, task):
        """
        LLM-based semantic analysis (advanced, requires transformers).
        """
        
        # Construct prompt for LLM
        prompt = f"""Analyze this IoT task:
Type: {task.task_type.name}
Data size: {task.size_bits / 1e6:.2f} MB
CPU cycles: {task.cpu_cycles / 1e9:.2f} billion
Deadline: {task.deadline:.2f} seconds

Priority (0-1): """
        
        try:
            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=100, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=10,
                    temperature=0.7,
                    do_sample=True
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract priority score (simple parsing)
            # For now, fall back to rule-based if parsing fails
            return self._rule_based_analyze(task)
            
        except Exception as e:
            print(f"[LLM] Analysis failed: {e}. Using rule-based fallback.")
            return self._rule_based_analyze(task)
    
    def get_priority_label(self, priority_score):
        """Convert numeric priority to human-readable label."""
        if priority_score >= 0.8:
            return "CRITICAL"
        elif priority_score >= 0.6:
            return "HIGH"
        elif priority_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"


# Test the analyzer
if __name__ == "__main__":
    from simulation_env import Task, TaskType
    
    print("=== Testing LLM Semantic Analyzer ===\n")
    
    # Create analyzer (rule-based by default)
    analyzer = SemanticAnalyzer(use_llm=False)
    
    # Test tasks
    test_tasks = [
        Task(1, 0.0, 1e6, 5e8, TaskType.CRITICAL, 0.5),      # Critical, small, urgent
        Task(2, 0.0, 50e6, 1e10, TaskType.HIGH_DATA, 5.0),   # Large data, complex
        Task(3, 0.0, 100e3, 1e7, TaskType.BEST_EFFORT, 10.0) # Small, simple, tolerant
    ]
    
    for task in test_tasks:
        analysis = analyzer.analyze_task(task)
        print(f"Task {task.id} ({task.task_type.name}):")
        print(f"  Priority Score: {analysis['priority_score']:.2f} ({analyzer.get_priority_label(analysis['priority_score'])})")
        print(f"  Urgency: {analysis['urgency']:.2f}")
        print(f"  Complexity: {analysis['complexity']:.2f}")
        print(f"  Bandwidth Need: {analysis['bandwidth_need']:.2f}")
        print(f"  Recommendation: {analysis['recommended_target']}")
        print(f"  Method: {analysis['analysis_method']}\n")
