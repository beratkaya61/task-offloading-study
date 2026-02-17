# Progress Article 01: Next-Generation IoT Task Offloading System Architecture

## 1. Introduction
Modern IoT application environments (e.g., autonomous vehicles, smart cities, and industrial drones) require high computing power with low latency. However, IoT devices are inherently limited by battery capacity and processing speed. **Task Offloading** addresses this by delegating heavy computations to more powerful layers: **Edge Servers** (low latency, medium power) and the **Cloud** (high latency, infinite power).

## 2. Our 3-Layer Architecture
Our simulation implements a dynamic 3-layer offloading model:

| Layer | Component | Mobility | Characteristic |
| :--- | :--- | :--- | :--- |
| **Local** | IoT Device | High (Vehicle) | Battery limited, local CPU only. |
| **Edge** | Edge Server | Static (Building) | Distributed infrastructure, fast response. |
| **Cloud** | Cloud Server | Virtual/Remote | Massive resources, higher delay (RTT). |

## 3. Why This Approach?
We use a **Hybrid Offloading Pipeline**:
1. **Semantic Awareness**: Before making a networking decision, we analyze *what* the task is (e.g., is it a critical safety alert or just a background log?).
2. **Dynamic Decision Making**: Decisions are not static. They change every second based on signal strength (Shannon), server load (Queueing), and remaining battery life.

## 4. Current Progress
- [x] Multi-agent simulation with SimPy.
- [x] Real-time visualization with PyGame.
- [x] Dynamic mobility models for vehicles.
- [x] Integrated infrastructure (3 Edge + 1 Cloud).

---
*Author: Antigravity AI Implementation System*
*Date: February 2026*
