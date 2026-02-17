# Implementation Plan - AI Offloading & Reinforcement Learning

This plan covers the integration of LLM-based semantic analysis and the upcoming transition to Deep Reinforcement Learning (PPO).

## 1. Status: Completed Enhancements
- [x] **Semantic Analyzer**: Classification of tasks based on priority/complexity.
- [x] **Professional GUI (Dashboard)**: 
    - Mission Control layout with Glassmorphism.
    - AI Decision Feed with mathematical traces (Shannon, DVFS, SNR).
    - Scientific Knowledge Booklet for academic context.
- [x] **Dynamic Mathematical Models**: Real-time calculated communication/computation costs.
- [x] **Academic Progress Logs**: Makale formatında dokümantasyon (`src/progress/`).
- [x] **RL Theory Document**: `04_RL_and_Reward_Shaping.md` created.

## 2. Next Phase: PPO Reinforcement Learning
Integrating an intelligent agent using Stable-Baselines3.

### [Component] RL Environment - `src/rl_env.py`
- [x] Custom Gymnasium environment defined.
- [x] State space [SNR, Size, Cycles, Battery, Load] normalized.
- [x] Action space [Local, Edge, Cloud] discrete.

### [Component] Training Script - `src/train_agent.py`
- [x] PPO training harness created.
- [x] Integration with `SB3` instantiated.

### [Component] LLM Reward Shaping
- [x] Semantic analysis scores integrated into Reward/Penalty logic.
- [x] Priority-based deadline penalties implemented.


## 3. Phase 3: GUI & Explainability Overhaul
Significant layout redesign and advanced interactivity.

### [Component] Layout Expansion - `src/gui.py`
- [x] **Dual-Sidebar Layout**: Expand `SCREEN_WIDTH` to 1800 to accommodate a new **Left Methodology Panel**.
- [x] **Methodology Expansion**: Detailed technical explanations for Shannon, Energy, SNR, and AI Logic with purpose/benefits.
- [x] **UI Collision Fix**: Adjust y-offsets in `draw_legend` and `draw_knowledge_base` to prevent overlap.
- [x] **Legibility Upgrade**: Increase font sizes for methodology headers and descriptions.
- [x] **Granular Stats**: Breakdown "Offloaded" count by Cloud, Edge-0, Edge-1, etc.
- [x] **Node Health Metrics**: Visual health indicators per specific device and edge node.

### [Component] Interactivity & Visualization - `src/gui.py`
- [ ] **Map Zoom & Pan**: Implement `zoom_level` and `offset` logic for the central map.
- [ ] **Fixed Sidebars**: Ensure side panels remain static while the map scales.

### [Component] AI Semantic Feedback - `src/simulation_env.py`
- [ ] **PPO vs. LLM Clarification**: Rewrite feedback strings to clearly distinguish between **LLM Analysis** (Semantic Context) and **AI Recommendation** (PPO Optimization).

## 4. Verification Plan
1. **Layout**: Confirm both sidebars are visible and map is centered.
2. **Zoom**: Test mouse wheel zoom on the map.
3. **Stat Accuracy**: Verify per-edge offloading counts match simulation events.
4. **Health**: Check that specific overloaded nodes show red/offline status.

---
*Date: February 17, 2026*
