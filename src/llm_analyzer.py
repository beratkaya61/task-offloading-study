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
    
    def analyze_task(self, task, device_battery_pct=None, network_quality_pct=None, 
                     edge_load_pct=None, cloud_latency=None):
        """
        Analyze a task and return semantic metadata.
        
        Args:
            task: Task object with attributes (task_type, size_bits, cpu_cycles, deadline)
            device_battery_pct: Device battery percentage (0-100) - OPTION B: Context enrichment
            network_quality_pct: Network quality (0-100, 100=perfect) - OPTION B: Context enrichment
            edge_load_pct: Edge server load percentage (0-100) - OPTION B: Context enrichment
            cloud_latency: Cloud latency in seconds - OPTION B: Context enrichment
        
        Returns:
            dict with keys:
                - priority_score: float (0-1, higher = more important)
                - urgency: float (0-1, based on deadline)
                - complexity: float (0-1, based on cpu cycles)
                - bandwidth_need: float (0-1, based on data size)
                - recommended_target: str ("local", "edge", "cloud")
                - confidence: float (0-1, LLM confidence in recommendation)
        """
        
        # ✅ Track which method is used
        if self.use_llm:
            result = self._llm_analyze(task, device_battery_pct, network_quality_pct, 
                                     edge_load_pct, cloud_latency)
            # Increment LLM success counter
            self.llm_success_count += 1
        else:
            result = self._rule_based_analyze(task, device_battery_pct, network_quality_pct, 
                                           edge_load_pct, cloud_latency)
            # Increment rule-based counter
            self.rule_based_fallback_count += 1
        
        return result
    
    def _rule_based_analyze(self, task, device_battery_pct=None, network_quality_pct=None, 
                           edge_load_pct=None, cloud_latency=None):
        """
        Rule-based semantic analysis (fast, no LLM needed).
        Now with context-aware decision making (OPTION B).
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
        
        # ✅ OPTION B: Context-Aware Recommendation Logic
        confidence = 0.5  # Default confidence
        
        if device_battery_pct is not None and device_battery_pct < 10:
            # Critical battery: must use local
            recommended_target = "local"
            confidence = 0.95
            reason = f"Batarya kritik seviyede ({device_battery_pct:.1f}%). Local işleme zorunlu."
        elif network_quality_pct is not None and network_quality_pct < 20:
            # Poor network: avoid transmission
            recommended_target = "local"
            confidence = 0.90
            reason = f"Ağ kalitesi çok düşük ({network_quality_pct:.1f}%). Local işlem tercih."
        elif bandwidth_need > 0.7:
            # Large data
            if edge_load_pct is not None and edge_load_pct > 80:
                recommended_target = "cloud"
                confidence = 0.85
                reason = f"Büyük veri ({task.size_bits/1e6:.1f}MB) ve Edge yoğun ({edge_load_pct:.1f}%). Cloud seçildi."
            else:
                recommended_target = "edge"
                confidence = 0.80
                reason = f"Büyük veri ({task.size_bits/1e6:.1f}MB). Edge sunucusuna yolla."
        elif task.task_type.name == "CRITICAL":
            # Critical tasks: low latency
            if network_quality_pct is not None and network_quality_pct > 60:
                recommended_target = "edge"
                confidence = 0.90
                reason = f"CRITICAL görev + iyi ağ ({network_quality_pct:.1f}%). Edge en hızlı."
            else:
                recommended_target = "edge"
                confidence = 0.80
                reason = "CRITICAL görev. Düşük gecikme için Edge seçildi."
        elif complexity < 0.3 and (device_battery_pct is None or device_battery_pct > 30):
            # Simple task with good battery
            recommended_target = "local"
            confidence = 0.85
            reason = f"Düşük karmaşıklık ({complexity:.2f}) ve yeterli batarya. Local tasarrufu."
        else:
            # Default to edge for balance
            recommended_target = "edge"
            confidence = 0.70
            reason = f"Denge çözümü: Edge sunucusu seçildi."
        
        return {
            "priority_score": round(priority_score, 2),
            "urgency": round(deadline_urgency, 2),
            "complexity": round(complexity, 2),
            "bandwidth_need": round(bandwidth_need, 2),
            "recommended_target": recommended_target,
            "confidence": round(confidence, 2),  # ✅ NEW: Confidence score
            "analysis_method": "Semantic Analyzer with Context Awareness",
            "reason": reason,
            "raw_stats": {
                "size_mb": round(task.size_bits / 1e6, 2),
                "cpu_ghz": round(task.cpu_cycles / 1e9, 2),
                "battery_pct": round(device_battery_pct, 1) if device_battery_pct else "Unknown",
                "network_quality": round(network_quality_pct, 1) if network_quality_pct else "Unknown",
                "edge_load": round(edge_load_pct, 1) if edge_load_pct else "Unknown"
            }
        }
    
    def _llm_analyze(self, task, device_battery_pct=None, network_quality_pct=None, 
                    edge_load_pct=None, cloud_latency=None):
        """
        LLM-based semantic analysis using Few-Shot Prompting (instruction-tuned model).
        Now with context-aware decision making (OPTION B).
        
        Few-Shot Examples help the model understand the expected format and reasoning.
        """
        
        # Few-Shot Examples (talimat izletme örnekleri) - Now with context
        few_shot_examples = """
[EXAMPLE 1]
Input: Task Type: CRITICAL, Size: 1.50 MB, CPU: 0.50 GHz, Deadline: 0.50 seconds
Context: Battery: 85%, Network: 80%, Edge Load: 40%
Analysis:
- Priority Score: 0.85 (CRITICAL tasks need immediate response)
- Urgency: 0.95 (Very short deadline, must process quickly)
- Complexity: 0.05 (Low CPU requirement)
- Bandwidth Need: 0.19 (Small data size)
- Recommendation: EDGE (Critical tasks benefit from low latency of edge servers + good network)
- Confidence: 0.95 (Clear decision: critical + good conditions)
- Reason: CRITICAL priority with ultra-short deadline demands immediate edge processing. Good network and low edge load support this decision.

[EXAMPLE 2]
Input: Task Type: HIGH_DATA, Size: 50.00 MB, CPU: 10.00 GHz, Deadline: 5.00 seconds
Context: Battery: 50%, Network: 30%, Edge Load: 90%
Analysis:
- Priority Score: 0.65 (High data workload, moderate urgency)
- Urgency: 0.17 (Reasonable deadline allows flexibility)
- Complexity: 1.00 (Very high CPU demand exceeds edge capacity)
- Bandwidth Need: 0.63 (Large data transfer with poor network)
- Recommendation: CLOUD (Large data + poor network + edge overloaded = CLOUD best option)
- Confidence: 0.85 (Clear decision: conflicting constraints favor cloud)
- Reason: High computational complexity and large data size (50MB) exceed edge processing capacity. Poor network (30%) makes CLOUD safer despite higher latency. Edge overloaded (90%).

[EXAMPLE 3]
Input: Task Type: BEST_EFFORT, Size: 0.10 MB, CPU: 0.01 GHz, Deadline: 10.00 seconds
Context: Battery: 8%, Network: 50%, Edge Load: 20%
Analysis:
- Priority Score: 0.25 (Low priority background task)
- Urgency: 0.09 (Long deadline, delay tolerant)
- Complexity: 0.00 (Minimal computation)
- Bandwidth Need: 0.01 (Negligible data transfer)
- Recommendation: LOCAL (Critical battery level overrides all other factors)
- Confidence: 0.95 (Battery < 10% = mandatory LOCAL)
- Reason: Trivial computational load BUT battery critically low (8%). Must use LOCAL to preserve device power. Freeing network resources not a concern here.
"""
        
        # ✅ OPTION B: Context-aware prompt
        context_str = ""
        if device_battery_pct is not None:
            context_str += f"Battery: {device_battery_pct:.1f}%, "
        if network_quality_pct is not None:
            context_str += f"Network: {network_quality_pct:.1f}%, "
        if edge_load_pct is not None:
            context_str += f"Edge Load: {edge_load_pct:.1f}%, "
        if cloud_latency is not None:
            context_str += f"Cloud Latency: {cloud_latency:.2f}s"
        
        context_str = context_str.rstrip(", ") if context_str else "Standard conditions"
        
        # Construct few-shot prompt for TinyLlama (instruction-tuned)
        prompt = f"""You are an IoT Task Offloading Analyzer. Your job is to analyze tasks and recommend where they should execute (LOCAL device, EDGE server, or CLOUD).

{few_shot_examples}

[NEW TASK TO ANALYZE]
Input: Task Type: {task.task_type.name}, Size: {task.size_bits / 1e6:.2f} MB, CPU: {task.cpu_cycles / 1e9:.2f} GHz, Deadline: {task.deadline:.2f} seconds
Context: {context_str}

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
                return self._rule_based_analyze(task, device_battery_pct, network_quality_pct, 
                                               edge_load_pct, cloud_latency)
            
        except Exception as e:
            self.rule_based_fallback_count += 1
            print(f"[LLM] Exception during analysis: {e}. Using rule-based fallback.")
            return self._rule_based_analyze(task, device_battery_pct, network_quality_pct, 
                                           edge_load_pct, cloud_latency)
    
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
            
            # ✅ NEW: Extract or estimate confidence from LLM response
            confidence_match = re.search(r"Confidence:\s*([\d.]+)", analysis_text)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.75
            confidence = min(1.0, max(0.0, confidence))  # Clamp to [0, 1]
            
            return {
                "priority_score": round(priority_score, 2),
                "urgency": round(urgency, 2),
                "complexity": round(complexity, 2),
                "bandwidth_need": round(bandwidth_need, 2),
                "recommended_target": recommended_target,
                "confidence": round(confidence, 2),  # ✅ NEW: Confidence score
                "analysis_method": "TinyLlama (Instruction-Tuned) + Few-Shot Prompting",
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
