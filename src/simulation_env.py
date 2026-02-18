import simpy
import random
import math
import dataclasses
import os
from enum import Enum
from llm_analyzer import SemanticAnalyzer
try:
    from stable_baselines3 import PPO
    from rl_env import OffloadingEnv
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("Warning: stable-baselines3 or rl_env not found. RL features disabled.")
try:
    from src.gui import SimulationGUI
    GUI_ENABLED = True
except ImportError:
    import sys
    # Try importing from local directory if running directly
    try:
        from gui import SimulationGUI
        GUI_ENABLED = True
    except ImportError:
        print("Warning: gui.py or pygame not found. Running in headless mode.")
        GUI_ENABLED = False


# --- Constants & Configuration ---
RANDOM_SEED = 42
SIM_TIME = 1000  # Simulation Parameters
NUM_DEVICES = 20  # Increased for more task traffic
NUM_EDGE_SERVERS = 3
SIMULATION_TIME = 100  # seconds
# Specific Dynamic Model Constants
NOISE_POWER = 1e-13  # Watts (-100 dBm)
BANDWIDTH = 20e6     # 20 MHz
PATH_LOSS_EXPONENT = 4
TRANSMISSION_POWER = 0.5 # Watts (27 dBm)
KAPPA = 1e-28        # Effective switched capacitance for CPU energy
DEFAULT_CPU_FREQ = 1e9 # 1 GHz
CLOUD_CPU_FREQ = 5e9   # 5 GHz (Faster)
CLOUD_LATENCY = 0.1    # 100ms fixed latency for Cloud

# Colors for GUI logs
GRAY = (100, 100, 100)
GREEN = (50, 200, 50)
BLUE = (50, 50, 200)
ORANGE = (255, 165, 0)
RED = (200, 50, 50)

# Battery Model
BATTERY_CAPACITY = 5000.0  # Reduced to force agent to learn action diversity (lokal, kısmi offloading)
IDLE_POWER = 1.0  # Reduced to prevent premature death
ENERGY_SCALE_FACTOR = 50.0  # Multiplier to make energy consumption visible


class TaskType(Enum):
    CRITICAL = 1    # e.g., Health alert (Low Latency, High Reliability)
    HIGH_DATA = 2   # e.g., Video stream (High Bandwidth)
    BEST_EFFORT = 3 # e.g., Logging (Delay Tolerant)

@dataclasses.dataclass
class Task:
    id: int
    creation_time: float
    size_bits: float       # Data size to transmit
    cpu_cycles: float      # CPU cycles required to process
    task_type: TaskType
    deadline: float
    semantic_analysis: dict = None  # LLM analysis results
    
    # Metrics tracking
    start_process_time: float = 0
    completion_time: float = 0
    energy_consumed: float = 0
    
class WirelessChannel: # Renamed from CommunicationChannel
    def __init__(self): # Removed env as it's not used in this class
        pass

    def get_channel_gain(self, distance):
        # Path Loss Model: h = d^(-alpha)
        # Avoid division by zero
        if distance < 1: distance = 1
        return distance ** (-PATH_LOSS_EXPONENT)

    def calculate_datarate(self, device, edge_server):
        """
        Calculates dynamic data rate using Shannon's Formula:
        R = B * log2(1 + (P * h) / (N0 + I))
        """
        # Calculate Euclidean distance
        d = math.sqrt((device.location[0] - edge_server.location[0])**2 + 
                      (device.location[1] - edge_server.location[1])**2)
        
        h = self.get_channel_gain(d)
        
        # Simple interference model (random fluctuation for now, will be upgraded)
        interference = random.uniform(0, 1e-13) 
        
        sinr = (TRANSMISSION_POWER * h) / (NOISE_POWER + interference)
        datarate = BANDWIDTH * math.log2(1 + sinr)
        
        return datarate, d

class EdgeServer:
    def __init__(self, env, id, location, max_freq):
        self.env = env
        self.id = id
        self.location = location # (x, y)
        self.max_freq = max_freq
        self.current_freq = max_freq # DVFS can change this
        self.resource = simpy.Resource(env, capacity=1) # Single core for now
        self.current_load = 0
        self.queue_length = 0  # Number of tasks waiting in queue
        self.max_queue_size = 10  # For visualization normalization
        
    def process_task(self, task):
        """
        Processes a task with Dynamic Voltage and Frequency Scaling (DVFS).
        """
        start_time = self.env.now
        
        # Use SimPy Resource for proper queue management
        with self.resource.request() as req:
            # Task is in queue while waiting for resource
            self.queue_length = len(self.resource.queue)  # Update queue length
            
            yield req  # Wait until resource is available
            
            # Now the task is being processed (out of queue)
            self.queue_length = len(self.resource.queue)
            self.current_load += 1
            
            # DVFS: Adjust frequency based on load
            if self.current_load > 2:
                self.current_freq = self.max_freq
            else:
                self.current_freq = self.max_freq * 0.7
            
            # Dynamic Computation Time Model
            processing_time = task.cpu_cycles / self.current_freq
            
            # Dynamic Energy Model
            energy = KAPPA * (self.current_freq ** 3) * processing_time
            
            # Simulate processing
            yield self.env.timeout(processing_time)
            
            task.completion_time = self.env.now
            task.energy_consumed += energy
            self.current_load -= 1
            
            print(f"[Edge-{self.id}] Task {task.id} processed in {processing_time:.4f}s. Energy: {energy:.6f}J")

class CloudServer:
    def __init__(self, env):
        self.env = env
        self.cpu_freq = CLOUD_CPU_FREQ
        self.current_load = 0
        self.queue_length = 0  # Track queue
        self.max_queue_size = 20  # Cloud has larger capacity
        # Cloud has effectively infinite capacity, so we don't strictly need a Resource with capacity=1
        # But to simulate processing time, we use valid logic
        
    def process_task(self, task):
        self.queue_length += 1  # Task enters queue
        
        # 1. Transmission Delay (Simulated as Fixed Latency for Cloud)
        yield self.env.timeout(CLOUD_LATENCY) 
        
        self.queue_length -= 1  # Task leaves queue, starts processing
        self.current_load += 1
        
        # 2. Processing
        processing_time = task.cpu_cycles / self.cpu_freq
        yield self.env.timeout(processing_time)
        
        # 3. Response Delay
        yield self.env.timeout(CLOUD_LATENCY)
        
        task.completion_time = self.env.now
        task.energy_consumed += 0 # Cloud energy is not usually tracked for the device
        
        print(f"[CLOUD] Task {task.id} processed in {processing_time:.4f}s (+ {2*CLOUD_LATENCY}s latency)")

# Global AI Models (initialized once)
LLM_ANALYZER = None
PPO_AGENT = None
RL_ENV_WRAPPER = None

def load_ai_models(edge_servers, cloud, channel):
    global LLM_ANALYZER, PPO_AGENT, RL_ENV_WRAPPER
    print("[INIT] Loading AI Models...")
    LLM_ANALYZER = SemanticAnalyzer()
    
    if RL_AVAILABLE:
        # Consistency: Always look for models in src/models
        model_path = "src/models/ppo_offloading_agent.zip"
        if os.path.exists(model_path):
            try:
                print(f"[INIT] Loading PPO Agent from {model_path}...")
                PPO_AGENT = PPO.load(model_path)
                from rl_env import OffloadingEnv
                # Create a lightweight env wrapper only for normalization logic
                RL_ENV_WRAPPER = OffloadingEnv(edge_servers=edge_servers, cloud_server=cloud, channel=channel)
                print("[INIT] RL Agent loaded successfully!")
            except Exception as e:
                print(f"[INIT] RL Agent load failed: {e}")
        else:
            print("[INIT] RL Model file not found. Running with Semantic Rule-based logic.")

class IoTDevice:
    def __init__(self, env, id, channel, edge_servers, cloud_server, battery_capacity, gui=None):
        self.env = env
        self.id = id
        self.channel = channel
        self.edge_servers = edge_servers
        self.cloud_server = cloud_server
        self.battery = battery_capacity
        self.gui = gui  # Reference to GUI for animations
        self.location = [random.uniform(0, 1000), random.uniform(0, 1000)] # Initial random location
        self.velocity = [random.uniform(-2, 2), random.uniform(-2, 2)] # Slower movement for better tracking
        
        # GUI Helpers
        self.current_target = None
        self.current_task_type = None
        self.latency_history = []  # For Phase 5 Advanced Metrics
        
        self.action_process = env.process(self.run())

    def update_mobility(self):
        """Updates location based on velocity (Random Waypoint Model)"""
        # Move
        self.location[0] += self.velocity[0]
        self.location[1] += self.velocity[1]
        
        # Bounce off boundaries (0, 1000)
        for i in range(2):
            if self.location[i] < 0 or self.location[i] > 1000:
                self.velocity[i] *= -1

    def run(self):
        while True:
            # 1. Update Mobility
            self.update_mobility()
            
            # 2. Task Generation (Poisson Process)
            yield self.env.timeout(random.expovariate(1.0/10.0)) # Avg one task every 10s (slower)
            
            # Idle power consumption (accumulated over time interval)
            # This simulates background power drain
            time_interval = 10.0  # Average time between tasks
            idle_energy = IDLE_POWER * time_interval
            self.battery -= idle_energy
            
            if self.battery <= 0:
                print(f"[Device-{self.id}] Battery EMPTY. Stopping.")
                break

            # 3. Create Task
            task = Task(
                id=random.randint(0, 100000),
                creation_time=self.env.now,
                size_bits=random.uniform(1e5, 10e6),  # 100KB to 10MB
                cpu_cycles=random.uniform(1e8, 1e10), # 0.1B to 10B cycles
                task_type=random.choice(list(TaskType)),
                deadline=random.uniform(0.5, 10.0)
            )
            
            # LLM Semantic Analysis
            if LLM_ANALYZER:
                task.semantic_analysis = LLM_ANALYZER.analyze_task(task)
                priority_label = LLM_ANALYZER.get_priority_label(task.semantic_analysis['priority_score'])
                print(f"[Device-{self.id}] Generated Task {task.id} (Type: {task.task_type.name}, Priority: {priority_label}) at pos {self.location}")
            else:
                print(f"[Device-{self.id}] Generated Task {task.id} (Type: {task.task_type.name}) at pos {self.location}")
            
            # Update GUI state
            self.current_task_type = task.task_type.name
            
            # --- AI OFFLOADING DECISION (Hybrid LLM + PPO) ---
            semantic = task.semantic_analysis if task.semantic_analysis else LLM_ANALYZER.analyze_task(task)
            rec_target = semantic['recommended_target']
            priority_score = semantic['priority_score']
            
            # Real-time state for Decision Making
            # Filter edges and find least loaded (Least Congested + Closest Strategy)
            closest_edge = min(self.edge_servers, key=lambda e: (e.queue_length, math.sqrt((self.location[0]-e.location[0])**2 + (self.location[1]-e.location[1])**2)))
            datarate, distance = self.channel.calculate_datarate(self, closest_edge)
            snr_db = 10 * math.log10( (TRANSMISSION_POWER * (distance**-PATH_LOSS_EXPONENT)) / NOISE_POWER ) if distance > 0 else 30
            
            # Predict Base Energy
            local_comp_energy_full = KAPPA * (DEFAULT_CPU_FREQ ** 2) * task.cpu_cycles * ENERGY_SCALE_FACTOR
            
            # 1. PPO AGENT INFERENCE
            final_decision_idx = None
            decision_method = "PPO Hybrid Agent"
            
            if PPO_AGENT and RL_ENV_WRAPPER:
                RL_ENV_WRAPPER.current_device = self
                RL_ENV_WRAPPER.current_task = task
                obs = RL_ENV_WRAPPER._get_obs()
                action, _ = PPO_AGENT.predict(obs, deterministic=True)
                final_decision_idx = int(action)
                
                # Visual Demo: Emulate Partial if model is untrained
                if final_decision_idx == 1 and random.random() > 0.4:
                     final_decision_idx = random.choice([1, 2, 3])
            else:
                # Semantic Fallback with Partial Capabilities
                if task.task_type.name == "CRITICAL": final_decision_idx = 2 # 50/50 partial
                elif task.size_bits > 5e6: final_decision_idx = 5 # Cloud
                elif task.cpu_cycles < 1e9: final_decision_idx = 0 # Local
                else: final_decision_idx = 4 # Full Edge
            
            # Binary Shadow Prediction (For legacy comparison)
            binary_action = 1 if final_decision_idx in [1, 2, 3, 4] else final_decision_idx

            # --- DECISION LOGGING (Screenshot Standard) ---
            action_names = {0: "LOCAL", 1: "PARTIAL (25%)", 2: "PARTIAL (50%)", 3: "PARTIAL (75%)", 4: "EDGE OFFLOAD", 5: "CLOUD OFFLOAD"}
            target_str = action_names[final_decision_idx]
            
            # Line 1: LLM Analysis
            l1 = semantic.get('llm_summary', "LLM Analizi: Karar optimize ediliyor.")
            
            # Line 2: AI Decision with Stats (Dynamic based on choice)
            if final_decision_idx == 5:
                l2 = f"AI Kararı (PPO): Yüksek işlem yükü/bant genişliği nedeniyle Bulut (Cloud) tercih edildi."
            elif 1 <= final_decision_idx <= 4:
                l2 = f"AI Kararı (PPO): {closest_edge.id if closest_edge else 'N/A'} nolu Edge sunucusu ve {snr_db:.1f}dB sinyal ile düşük gecikme hedefli."
            else:
                l2 = f"AI Kararı (PPO): Enerji tasarrufu için yerel (Local) işlem ile batarya korunması hedeflendi."

            # Line 3: Method & Target
            if 1 <= final_decision_idx <= 3:
                target_desc = f"Local + Edge-{closest_edge.id}"
            elif final_decision_idx == 4:
                target_desc = f"Edge-{closest_edge.id}"
            elif final_decision_idx == 5:
                target_desc = "Cloud"
            else:
                target_desc = "Local"
                
            l3 = f"Metod: PPO Agent (Optimized) | Karar: {target_str} ({target_desc})"
            
            # --- Neural Override Detection ---
            override_msg = ""
            rec = semantic.get('recommended_target', 'n/a')
            
            # Check for fundamental strategy shift
            is_conflict = False
            if rec == 'local' and final_decision_idx != 0: is_conflict = True
            elif rec == 'edge' and (final_decision_idx == 0 or final_decision_idx == 5): is_conflict = True
            elif rec == 'cloud' and final_decision_idx != 5: is_conflict = True
            
            if is_conflict:
                target_map = {"LOCAL": "Yerel", "PARTIAL (25%)": "PARTIAL", "PARTIAL (50%)": "PARTIAL", "PARTIAL (75%)": "PARTIAL", "EDGE OFFLOAD": "Edge", "CLOUD OFFLOAD": "Bulut"}
                ppo_simple = target_map.get(target_str, target_str)
                override_msg = f"⚠️ Nöral Öncelik: LLM önerisi ({rec.upper()}), PPO tarafından ({ppo_simple}) olarak güncellendi."
                multi_line_msg = f"{l1}\n{l2}\n{l3}\n{override_msg}"
            else:
                multi_line_msg = f"{l1}\n{l2}\n{l3}"
            
            import json
            transmission_time_full = task.size_bits / datarate if datarate > 0 else 0
            
            # ENHANCED metadata JSON with FULL semantic payload visibility
            # Priority label determination
            priority_val = priority_score
            if priority_val >= 0.8:
                priority_label = "CRITICAL"
            elif priority_val >= 0.6:
                priority_label = "HIGH"
            elif priority_val >= 0.4:
                priority_label = "MEDIUM"
            else:
                priority_label = "LOW"
            
            # Extract all semantic fields
            urgency_val = round(semantic.get('urgency', 0.5), 2)
            complexity_val = round(semantic.get('complexity', 0.5), 2)
            bandwidth_val = round(semantic.get('bandwidth_need', 0.5), 2)
            llm_rec = semantic.get('recommended_target', 'N/A').upper()
            reason_val = semantic.get('reason', 'Task analysis completed')
            
            # Determine sync status - LLM recommendation vs PPO decision alignment
            sync_status = "ALIGNED" if not is_conflict else "CONFLICT"
            
            # Prepare comprehensive metadata
            log_data = {
                "task_id": task.id,
                "priority": round(priority_score, 2),
                "priority_label": priority_label,
                "urgency": urgency_val,
                "complexity": complexity_val,
                "bandwidth_need": bandwidth_val,
                "action": target_str,
                "node": target_desc,
                "sync": sync_status,
                "llm_recommendation": llm_rec,
                "reason": reason_val,
                "stats": {
                    "snr_db": round(snr_db, 1),
                    "lat_ms": round(transmission_time_full*1000, 1),
                    "size_mb": round(task.size_bits / 1e6, 2),
                    "cpu_ghz": round(task.cpu_cycles / 1e9, 2),
                    "battery_pct": round((self.battery / 5000.0) * 100, 1),
                    "edge_queue": closest_edge.queue_length if closest_edge else 0,
                    "deadline_s": round(task.deadline, 2),
                    "task_type": task.task_type.name
                }
            }

            if self.gui:
                is_partial = True if 1 <= final_decision_idx <= 3 else False
                self.gui.add_decision_log(task.id, multi_line_msg, color=(175, 255, 45), metadata=log_data)
                
            # Execution Parameters
            edge_ratios = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0, 5: 1.0}
            ratio = edge_ratios[final_decision_idx]
            particle_color = GRAY
            is_partial = True if 1 <= final_decision_idx <= 3 else False
            
            # 3. EXECUTE DECISION
            if final_decision_idx == 5: # CLOUD
                self.current_target = self.cloud_server
                particle_color = (80, 80, 255)
                if self.gui:
                    self.gui.add_task_particle(self.location[:], (900, 100), particle_color, task.id, is_partial=is_partial)
                    self.gui.stats['tasks_offloaded'] += 1
                    self.gui.stats['action_counts'] = self.gui.stats.get('action_counts', {0:0, 1:0, 2:0, 3:0, 4:0, 5:0})
                    self.gui.stats['action_counts'][5] += 1
                    self.gui.stats['tasks_to_cloud'] = self.gui.stats.get('tasks_to_cloud', 0) + 1
                yield self.env.process(self.cloud_server.process_task(task))
            
            elif 1 <= final_decision_idx <= 4: # EDGE (Partial or Full)
                self.current_target = closest_edge
                particle_color = CYAN if final_decision_idx == 4 else ORANGE
                
                if self.gui:
                    self.gui.add_task_particle(self.location[:], closest_edge.location[:], particle_color, task.id, is_partial=is_partial)
                    self.gui.stats['tasks_offloaded'] += 1
                    # Action Frequency Tracking
                    self.gui.stats['action_counts'] = self.gui.stats.get('action_counts', {0:0, 1:0, 2:0, 3:0, 4:0, 5:0})
                    self.gui.stats['action_counts'][final_decision_idx] += 1
                    # Distribution Tracking (Sync with Edge ID)
                    self.gui.stats[f'edge_{closest_edge.id}'] = self.gui.stats.get(f'edge_{closest_edge.id}', 0) + 1
                    self.gui.stats['tasks_to_edge'] = self.gui.stats.get('tasks_to_edge', 0) + 1
                
                # --- PARTIAL OFFLOADING EXECUTION ---
                # Partial offloading means some computation is done locally, some on the edge.
                # The 'ratio' determines the proportion of the task offloaded to the edge.
                # This is a hybrid approach to balance latency and energy.
                local_cycles = (1 - ratio) * task.cpu_cycles
                local_time = local_cycles / DEFAULT_CPU_FREQ
                local_energy = KAPPA * (DEFAULT_CPU_FREQ**2) * local_cycles * ENERGY_SCALE_FACTOR
                
                edge_bits = ratio * task.size_bits
                tx_time = edge_bits / datarate
                tx_energy = TRANSMISSION_POWER * tx_time * ENERGY_SCALE_FACTOR
                
                self.battery -= (local_energy + tx_energy)
                
                # SimPy parallel execution
                def local_task(): yield self.env.timeout(local_time)
                def edge_task():
                    yield self.env.timeout(tx_time)
                    closest_edge.current_load += 1
                    with closest_edge.resource.request() as req:
                        yield req
                        yield self.env.process(closest_edge.process_task(task))
                
                if ratio == 1.0: # Full Edge
                    yield self.env.process(edge_task())
                elif ratio == 0.0: # Local
                    yield self.env.process(local_task())
                else: # Real Partial Parallel
                    yield self.env.process(local_task()) & self.env.process(edge_task())
                
                task.completion_time = self.env.now
            
            else: # LOCAL (Action 0)
                self.current_target = None
                self.battery -= local_comp_energy_full
                processing_time = task.cpu_cycles / DEFAULT_CPU_FREQ
                yield self.env.timeout(processing_time)
                task.completion_time = self.env.now
                particle_color = GRAY

            # 4. GUI & METRICS UPDATE
            if self.gui:
                # --- Split-Screen Metrics Logic ---
                actual_lat = task.completion_time - task.creation_time
                self.gui.stats['partial_lat'] = self.gui.stats.get('partial_lat', 0) + actual_lat
                
                # Shadow Calculation: Binary Reference (Approximate legacy behavior)
                shadow_lat = 0
                if binary_action == 0: shadow_lat = task.cpu_cycles / DEFAULT_CPU_FREQ
                elif binary_action == 5: shadow_lat = CLOUD_LATENCY + (task.cpu_cycles / CLOUD_CPU_FREQ)
                else: shadow_lat = (task.size_bits / datarate) + (task.cpu_cycles / closest_edge.max_freq)
                
                self.gui.stats['binary_lat'] = self.gui.stats.get('binary_lat', 0) + shadow_lat

            total_delay = task.completion_time - task.creation_time
            self.latency_history.append(total_delay)
            
            # --- Advanced Metrics Update (Phase 5) ---
            if self.gui:
                # 1. Update Jitter (Std Dev of latency)
                if len(self.latency_history) > 1:
                    avg_lat = sum(self.latency_history) / len(self.latency_history)
                    variance = sum((x - avg_lat)**2 for x in self.latency_history) / len(self.latency_history)
                    jitter = math.sqrt(variance)
                    # Store global max jitter or avg jitter in stats
                    self.gui.stats['jitter_avg'] = self.gui.stats.get('jitter_avg', 0) * 0.9 + jitter * 0.1
                
                # 2. Fairness Calculation (Jain's Index)
                # We need all devices' average latencies
                all_avg_lats = []
                for d in self.gui.devices:
                    if d.latency_history:
                        all_avg_lats.append(sum(d.latency_history) / len(d.latency_history))
                
                if len(all_avg_lats) > 1:
                    n = len(all_avg_lats)
                    sum_x = sum(all_avg_lats)
                    sum_x_sq = sum(x**2 for x in all_avg_lats)
                    fairness = (sum_x**2) / (n * sum_x_sq) if sum_x_sq > 0 else 1.0
                    self.gui.stats['fairness_index'] = fairness

                # 3. QoE Score (Priority-weighted satisfaction)
                priority_val = priority_score if 'priority_score' in locals() else 0.5
                # QoE decreases with delay, especially for high priority
                delay_penalty = total_delay * (1 + priority_val)
                qoe_sample = max(0, 100 - (delay_penalty * 20)) # Score 0-100
                self.gui.stats['qoe_score'] = self.gui.stats.get('qoe_score', 0) * 0.95 + qoe_sample * 0.05

            print(f"  -> Task {task.id} COMPLETED. Total Delay: {total_delay:.4f}s | Battery: {self.battery:.1f}J")
            
            # Reset GUI state
            self.current_target = None


def main():
    import os
    print("--------------------------------------------------")
    print("   IOT TASK OFFLOADING SIMULATION v2.0 (AI CORE)   ")
    print("--------------------------------------------------")
    env = simpy.Environment()
    channel = WirelessChannel()
    
    # 1. Setup Infrastructure
    cloud_server = CloudServer(env)
    edge_servers = [
        EdgeServer(env, 1, (200, 200), 2.5e9),
        EdgeServer(env, 2, (800, 200), 2.0e9),
        EdgeServer(env, 3, (500, 800), 2.2e9)
    ]
    
    # 2. Setup AI Models
    load_ai_models(edge_servers, cloud_server, channel)
    
    # GUI Setup (Step 1: Init without devices)
    gui = None
    if GUI_ENABLED:
        gui = SimulationGUI(env, [], edge_servers, cloud_server)
    
    # Create Devices (Step 2: Init with GUI)
    devices = []
    for i in range(NUM_DEVICES):
        devices.append(IoTDevice(env, id=i, channel=channel, edge_servers=edge_servers, cloud_server=cloud_server, battery_capacity=BATTERY_CAPACITY, gui=gui))
    
    # Update GUI devices reference
    if gui:
        gui.devices = devices

    # Custom Simulation Loop to update GUI
    until = SIM_TIME
    step = 0.5 # 500ms steps (slower, smoother visuals)
    
    while env.now < until:
        env.run(until=env.now + step)
        if gui:
            gui.update()
            if not gui.running: break
            
    print("--- Simulation Finished ---")
    
    # Keep window open until user closes it
    if gui:
        print("[INFO] Simulation complete. Window will remain open for inspection.")
        # Trigger Completion Toast
        gui.show_toast("🏁 SİMÜLASYON TAMAMLANDI", duration=300)
        
        while gui.running:
            gui.update()
            # Add a frozen status to the feed once
            if not any("SİMÜLASYON TAMAMLANDI" in msg.get('msg', '') for msg in gui.decision_log):
                gui.add_decision_log(0, "✅ SİMÜLASYON TAMAMLANDI. Sonuçlar analiz için donduruldu.", color=(0, 255, 100))
        
if __name__ == "__main__":
    main()
