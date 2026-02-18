# Progress Article 05: Semantic Decision Feed & Metadata Payload

## Overview

The **Semantic Decision Feed** is a real-time monitoring panel that shows how the PPO agent makes offloading decisions. Each decision includes rich metadata that explains:

- What the LLM recommended
- What the PPO agent actually chose
- Why they aligned or conflicted
- All relevant metrics for that decision

---

## Metadata Fields Explanation

### Core Decision Fields

| Field                | Type        | Description                                                                                                           |
| -------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------- |
| `task_id`            | Integer     | Unique identifier of the task                                                                                         |
| `priority`           | Float (0-1) | Overall priority score: `0=LOW`, `0.5=MEDIUM`, `0.75=HIGH`, `1.0=CRITICAL`                                            |
| `priority_label`     | String      | Human-readable priority: `"CRITICAL"`, `"HIGH"`, `"MEDIUM"`, `"LOW"`                                                  |
| `urgency`            | Float (0-1) | Time urgency based on deadline: shorter deadline → higher urgency                                                     |
| `complexity`         | Float (0-1) | Task complexity based on CPU cycles: more cycles → higher complexity                                                  |
| `bandwidth_need`     | Float (0-1) | Data transfer requirement: larger data → higher need                                                                  |
| `action`             | String      | PPO decision: `"LOCAL"`, `"PARTIAL (25%)"`, `"PARTIAL (50%)"`, `"PARTIAL (75%)"`, `"EDGE OFFLOAD"`, `"CLOUD OFFLOAD"` |
| `node`               | String      | Target compute node: `"Local"`, `"Local + Edge-1"`, `"Edge-2"`, `"Cloud"`                                             |
| `sync`               | String      | **LLM↔PPO Alignment Status**: `"LLM↔PPO Aligned"` or `"LLM↔PPO Conflict"`                                             |
| `llm_recommendation` | String      | What LLM suggested: `"LOCAL"`, `"EDGE"`, `"CLOUD"`                                                                    |
| `reason`             | String      | Human-readable explanation of the decision                                                                            |

### Performance Statistics (stats)

| Field         | Unit    | Description                                      |
| ------------- | ------- | ------------------------------------------------ |
| `snr_db`      | dB      | Signal-to-Noise Ratio (wireless channel quality) |
| `lat_ms`      | ms      | Predicted transmission latency                   |
| `size_mb`     | MB      | Task data size                                   |
| `cpu_ghz`     | GHz     | Computational requirement                        |
| `battery_pct` | %       | Device battery remaining                         |
| `edge_queue`  | Count   | Number of tasks waiting in Edge server queue     |
| `deadline_s`  | seconds | Task deadline constraint                         |
| `task_type`   | Enum    | `"CRITICAL"`, `"HIGH_DATA"`, `"BEST_EFFORT"`     |

---

## Decision Flow

```
1. Task Generated
   ↓
2. LLM Semantic Analysis
   ├─ Analyzes: priority, urgency, complexity, bandwidth_need
   └─ Outputs: recommended_target (LOCAL/EDGE/CLOUD)
   ↓
3. PPO Agent Decision
   ├─ Observes: [SNR, task_size, cpu_cycles, battery%, edge_load]
   └─ Outputs: action (0-5: Local to Cloud)
   ↓
4. Sync Check
   ├─ If PPO action matches LLM recommendation → "LLM↔PPO Aligned" ✓
   └─ If conflict detected → "LLM↔PPO Conflict" ⚠️
   ↓
5. Render Decision Feed
   └─ Display full metadata in GUI panel
```

---

## Sync Status Explanation

### LLM↔PPO Aligned ✓

- The PPO agent learned that the LLM's recommendation is **strategically correct**
- Example: Task is CRITICAL + short deadline → LLM says EDGE → PPO chooses EDGE ✓
- This indicates the agent is learning human-like decision patterns

### LLM↔PPO Conflict ⚠️

- The PPO agent chose differently than LLM recommendation
- Example: Task is CRITICAL + short deadline → LLM says EDGE → PPO chooses CLOUD ⚠️
- This can happen because:
  1. **Edge queue is full** → Cloud is faster despite LLM preference
  2. **Battery is critically low** → PPO offloads to preserve power
  3. **Signal strength is too weak** → Can't reliably reach Edge
  4. **PPO is still exploring** → May choose suboptimal actions during training

---

## Example Metadata Output

```json
{
  "task_id": 31425,
  "priority": 0.33 [LOW],
  "urgency": 0.72,
  "complexity": 0.07,
  "bandwidth_need": 0.1,
  "action": "PARTIAL (75%)" → "Local + Edge-1",
  "sync": "LLM↔PPO Aligned",
  "llm_recommendation": "EDGE",
  "reason": "Task type BEST_EFFORT with 2.5s deadline. Local + partial offloading balances latency and battery. 25% local processing preserves battery (45%) while Edge handles 75% for speed.",
  "stats": {
    "snr_db": 18.5,
    "lat_ms": 234.2,
    "size_mb": 0.95,
    "cpu_ghz": 2.15,
    "battery_pct": 45.0,
    "edge_queue": 2,
    "deadline_s": 2.5,
    "task_type": "BEST_EFFORT"
  }
}
```

---

## Key Insights from Metadata

### Why certain patterns emerge:

1. **High `urgency` + Low `battery%`** → Agent chooses Local/Partial (preserve power)
2. **High `complexity`** → Agent prefers Cloud (more compute power)
3. **High `edge_queue`** → Agent switches to Cloud (avoid queue delay)
4. **CRITICAL `task_type`** → Agent ignores battery, chooses fastest option
5. **Alignment trends** → Over time, PPO learns LLM's strategy → more "Aligned" decisions

---

## References

- Battery-aware offloading: Thresholds set at battery_pct < 30% for aggressive Local/Partial strategy
- Reward shaping coefficients: α=20 (latency), β=2 (energy), γ=50 (deadline penalty)
- LLM analysis method: Semantic Analyzer using rule-based classification (fallback to transformers if available)
