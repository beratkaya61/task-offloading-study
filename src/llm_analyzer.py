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
    
    def __init__(self, use_llm=False, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize the analyzer.
        
        Args:
            use_llm: Whether to use LLM-based analysis (requires transformers)
            model_name: Hugging Face model to use (default: TinyLlama - instruction-tuned and fast)
        """
        self.use_llm = use_llm and LLM_AVAILABLE
        self.model = None
        self.tokenizer = None
        self.llm_success_count = 0
        self.rule_based_fallback_count = 0
        
        if self.use_llm:
            try:
                print(f"[LLM] Loading model: {model_name}...")
                # Use TinyLlama - instruction-tuned (talimat izleyen) model
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                
                # Set pad token for batch processing
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                print(f"[LLM] Model loaded successfully!")
                print(f"[LLM] Using instruction-tuned model: {model_name}")
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
            llm_summary = f"LLM Analizi: Görev boyutu ({task.size_bits/1e6:.1f}MB) ve kritiklik seviyesi ({priority_score:.2f}) 'Cloud' için uygun."
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
        LLM-based semantic analysis using Few-Shot Prompting (instruction-tuned model).
        
        Few-Shot Examples help the model understand the expected format and reasoning.
        """
        
        # Few-Shot Examples (talimat izletme örnekleri)
        few_shot_examples = """
[EXAMPLE 1]
Input: Task Type: CRITICAL, Size: 1.50 MB, CPU: 0.50 GHz, Deadline: 0.50 seconds
Analysis:
- Priority Score: 0.85 (CRITICAL tasks need immediate response)
- Urgency: 0.95 (Very short deadline, must process quickly)
- Complexity: 0.05 (Low CPU requirement)
- Bandwidth Need: 0.19 (Small data size)
- Recommendation: EDGE (Critical tasks benefit from low latency of edge servers)
- Reason: CRITICAL priority with ultra-short deadline demands immediate edge processing to minimize response time and ensure patient safety.

[EXAMPLE 2]
Input: Task Type: HIGH_DATA, Size: 50.00 MB, CPU: 10.00 GHz, Deadline: 5.00 seconds
Analysis:
- Priority Score: 0.65 (High data workload, moderate urgency)
- Urgency: 0.17 (Reasonable deadline allows flexibility)
- Complexity: 1.00 (Very high CPU demand exceeds edge capacity)
- Bandwidth Need: 0.63 (Large data transfer, edge bandwidth constraint)
- Recommendation: CLOUD (Complex computation + large data requires cloud resources)
- Reason: High computational complexity and large data size exceed edge processing capacity. Cloud offers unlimited resources for parallel processing despite 200ms latency acceptable for 5-second deadline.

[EXAMPLE 3]
Input: Task Type: BEST_EFFORT, Size: 0.10 MB, CPU: 0.01 GHz, Deadline: 10.00 seconds
Analysis:
- Priority Score: 0.25 (Low priority background task)
- Urgency: 0.09 (Long deadline, delay tolerant)
- Complexity: 0.00 (Minimal computation)
- Bandwidth Need: 0.01 (Negligible data transfer)
- Recommendation: LOCAL (Minimal resource requirement, preserve edge/cloud for critical tasks)
- Reason: Trivial computational load and unlimited deadline make local processing ideal. Preserves device battery while freeing edge/cloud resources for high-priority tasks.
"""
        
        # Construct few-shot prompt for TinyLlama (instruction-tuned)
        prompt = f"""You are an IoT Task Offloading Analyzer. Your job is to analyze tasks and recommend where they should execute (LOCAL device, EDGE server, or CLOUD).

{few_shot_examples}

[NEW TASK TO ANALYZE]
Input: Task Type: {task.task_type.name}, Size: {task.size_bits / 1e6:.2f} MB, CPU: {task.cpu_cycles / 1e9:.2f} GHz, Deadline: {task.deadline:.2f} seconds

Analysis:
- Priority Score: """
        
        try:
            # Tokenize with attention to max_length
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=1500, 
                truncation=True,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=150,  # Increased for full analysis
                    temperature=0.3,      # Lower temperature for consistency
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the analysis part (after "Analysis:")
            if "Analysis:" in response:
                analysis_text = response.split("Analysis:")[-1]
            else:
                analysis_text = response
            
            # Try to parse structured response
            parsed = self._parse_llm_response(analysis_text, task)
            
            if parsed:
                self.llm_success_count += 1
                print(f"[LLM] ✓ Successful analysis for Task {task.id}")
                return parsed
            else:
                # Parsing failed, use rule-based fallback
                self.rule_based_fallback_count += 1
                print(f"[LLM] ✗ Parsing failed for Task {task.id}, using rule-based fallback")
                return self._rule_based_analyze(task)
            
        except Exception as e:
            self.rule_based_fallback_count += 1
            print(f"[LLM] Exception during analysis: {e}. Using rule-based fallback.")
            return self._rule_based_analyze(task)
    
    def _parse_llm_response(self, analysis_text, task):
        """
        Parse structured LLM response into our metadata format.
        Extracts key-value pairs from the analysis.
        """
        try:
            import re
            
            # Extract scores using regex
            priority_match = re.search(r"Priority Score:\s*([\d.]+)", analysis_text)
            urgency_match = re.search(r"Urgency:\s*([\d.]+)", analysis_text)
            complexity_match = re.search(r"Complexity:\s*([\d.]+)", analysis_text)
            bandwidth_match = re.search(r"Bandwidth Need:\s*([\d.]+)", analysis_text)
            recommendation_match = re.search(r"Recommendation:\s*(\w+)", analysis_text)
            reason_match = re.search(r"Reason:\s*(.+?)(?=\n|$)", analysis_text)
            
            # Validate extracted values
            priority_score = float(priority_match.group(1)) if priority_match else None
            urgency = float(urgency_match.group(1)) if urgency_match else None
            complexity = float(complexity_match.group(1)) if complexity_match else None
            bandwidth_need = float(bandwidth_match.group(1)) if bandwidth_match else None
            recommended_target = recommendation_match.group(1).lower() if recommendation_match else None
            reason = reason_match.group(1).strip() if reason_match else "LLM analysis provided."
            
            # Validate ranges (0-1)
            if not (0 <= priority_score <= 1 and 0 <= urgency <= 1 and 
                    0 <= complexity <= 1 and 0 <= bandwidth_need <= 1):
                return None
            
            # Validate recommendation
            if recommended_target not in ["local", "edge", "cloud"]:
                return None
            
            return {
                "priority_score": round(priority_score, 2),
                "urgency": round(urgency, 2),
                "complexity": round(complexity, 2),
                "bandwidth_need": round(bandwidth_need, 2),
                "recommended_target": recommended_target,
                "analysis_method": "TinyLlama (Instruction-Tuned) + Few-Shot Prompting",
                "llm_summary": f"LLM Analizi: {reason[:100]}...",
                "reason": reason,
                "raw_stats": {
                    "size_mb": round(task.size_bits / 1e6, 2),
                    "cpu_ghz": round(task.cpu_cycles / 1e9, 2)
                }
            }
        except Exception as e:
            print(f"[LLM] Parsing error: {e}")
            return None
    
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
    
    print("=== Testing LLM Semantic Analyzer (TinyLlama + Few-Shot) ===\n")
    
    # Create analyzer with LLM enabled
    analyzer = SemanticAnalyzer(use_llm=True, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
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
        print(f"  Recommendation: {analysis['recommended_target'].upper()}")
        print(f"  Method: {analysis['analysis_method']}")
        print(f"  Reason: {analysis['reason'][:80]}...\n")
    
    print(f"\n=== LLM Performance Stats ===")
    print(f"LLM Success Rate: {analyzer.llm_success_count} / {analyzer.llm_success_count + analyzer.rule_based_fallback_count}")
    print(f"Rule-Based Fallback Usage: {analyzer.rule_based_fallback_count} times")
