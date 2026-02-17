# Progress Article 03: Semantic Analysis and AI-Driven Offloading Decisions

## 1. The Core Problem: Beyond Greedy
Traditional offloading algorithms are "Greedy"—they always pick the closest server or the fastest link. This is inefficient because:
- A "Critical" task needs low latency regardless of energy.
- A "High Data" task needs bandwidth even if the server is far.
- A device with **Low Battery** should never execute locally.

## 2. LLM Semantic Analyzer
We integrated a **Semantic Analyzer** (powered by LLM/NLP concepts) to classify tasks:
- **CRITICAL**: Immediate action (e.g., brake assist).
- **HIGH_DATA**: High bandwidth requirement (e.g., video feed).
- **BEST_EFFORT**: Best-effort background processing (e.g., logs).

## 3. Decision Logic (The Semantic Brain)
The current decision process works as follows:
1. **Analyze**: Categorize task and determine its "Semantic Priority Score" (0-1).
2. **Observe**: Calculate Shannon datarate and server queue lengths.
3. **Decide**: 
   - If **Priority > 0.8** -> Force EDGE (Low Latency).
   - If **Battery < 15%** -> Force OFFLOAD (Power Saving).
   - If **Complexity > 0.8** -> **PARTIAL OFFLOADING** (Divide and Conquer).

## 4. Partial Offloading (Splitting)
Our system can split a single task into local and remote parts. This reduces local heat/energy while ensuring the deadline is met by offloading the majority of the heavy cycles to the Edge.

---
*Future Objective: Replacing rule-based logic with a Deep Reinforcement Learning (PPO) agent.*
