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
BATTERY_CAPACITY = 10000.0  # Joules (realistic for smartphone: ~10,000J = 2.7 Wh)
IDLE_POWER = 5.0  # Watts (idle consumption - increased for demo visibility)
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
            
            # --- AI OFFLOADING DECISION (Semantic + PPO Agent) ---
            # 1. Get LLM Semantic Analysis (Used for stats and fallback)
            semantic = task.semantic_analysis if task.semantic_analysis else LLM_ANALYZER.analyze_task(task)
            rec_target = semantic['recommended_target']
            priority_score = semantic['priority_score']
            
            # 2. Get Real-time Network Stats (Shannon Model)
            closest_edge = min(self.edge_servers, key=lambda e: math.dist(self.location, e.location))
            datarate, distance = self.channel.calculate_datarate(self, closest_edge)
            snr_db = 10 * math.log10( (TRANSMISSION_POWER * (distance**-PATH_LOSS_EXPONENT)) / NOISE_POWER ) if distance > 0 else 30
            
            # Predict Energy
            transmission_time = task.size_bits / datarate
            tx_energy_pred = TRANSMISSION_POWER * transmission_time * ENERGY_SCALE_FACTOR
            local_comp_energy_pred = KAPPA * (DEFAULT_CPU_FREQ ** 2) * task.cpu_cycles * ENERGY_SCALE_FACTOR
            
            # 3. PPO AGENT INFERENCE (If available)
            final_decision_idx = None
            decision_method = "Semantic Rules"
            
            if PPO_AGENT and RL_ENV_WRAPPER:
                # Prepare state vector
                RL_ENV_WRAPPER.current_device = self
                RL_ENV_WRAPPER.current_task = task
                obs = RL_ENV_WRAPPER._get_obs()
                
                # Predict action
                action, _ = PPO_AGENT.predict(obs, deterministic=True)
                final_decision_idx = action
                decision_method = "PPO Agent (Optimized)"
            else:
                # Fallback to Semantic Rules
                mapping = {"local": 0, "edge": 1, "cloud": 2}
                final_decision_idx = mapping.get(rec_target, 1)

            # 4. Final Decision Assignment
            decision_msg = ""
            final_target = None
            particle_color = GRAY
            
            # Battery-Aware Check Override
            battery_pct = (self.battery / BATTERY_CAPACITY) * 100
            if battery_pct < 15 and final_decision_idx == 0:
                final_decision_idx = 1 # Force offload on low battery
                decision_method += " + Battery Guard"

            if final_decision_idx == 2: # CLOUD
                target_type = "CLOUD"
                final_target = self.cloud_server
                particle_color = (80, 80, 255)
                decision_reason = (
                    f"🧠 LLM Analizi: Görev boyutu ({task.size_bits/1e6:.1f}MB) ve\n"
                    f"kritiklik seviyesi ({priority_score:.2f}) 'Bulut' için uygun.\n"
                    f"🤖 AI Kararı (PPO): Uzak bulut sunucusunda yüksek\n"
                    f"kapasite kullanımı optimize edildi.\n"
                    f"Metod: {decision_method} | Karar: CLOUD OFFLOAD"
                )
            elif final_decision_idx == 1: # EDGE
                target_type = "EDGE"
                final_target = closest_edge
                particle_color = (80, 255, 150)
                decision_reason = (
                    f"🧠 LLM Analizi: Bu görev ({task.task_type.name}) düşük\n"
                    f"gecikme gerektiriyor. Edge kullanımı öneriliyor.\n"
                    f"🤖 AI Kararı (PPO): Yakındaki Edge Node yük dengesi\n"
                    f"ve sinyal gücü ({snr_db:.1f}dB) için en iyi seçim.\n"
                    f"Metod: {decision_method} | Karar: EDGE OFFLOAD"
                )
            else: # LOCAL
                target_type = "LOCAL"
                final_target = None
                particle_color = GRAY
                decision_reason = (
                    f"🧠 LLM Analizi: Düşük karmaşıklıktaki görev yerel\n"
                    f"kaynaklarla batarya dostu şekilde çözülebilir.\n"
                    f"🤖 AI Kararı (PPO): Yerel işlemci (DVFS) kullanılarak\n"
                    f"enerji verimliliği ({local_comp_energy_pred:.1f}J) maximize edildi.\n"
                    f"Metod: {decision_method} | Karar: LOCAL EXECUTION"
                )

            # Log to GUI
            if self.gui:
                self.gui.add_decision_log(task.id, decision_reason, color=particle_color if final_target else GRAY)

            # Execute Decision
            if final_target == self.cloud_server:
                self.current_target = self.cloud_server
                if self.gui:
                    self.gui.add_task_particle(self.location[:], (900, 100), particle_color, task.id)
                    self.gui.stats['tasks_offloaded'] += 1
                    self.gui.stats['tasks_to_cloud'] += 1
                yield self.env.process(self.cloud_server.process_task(task))
            
            elif final_target == closest_edge:
                self.current_target = closest_edge
                edge_id = self.edge_servers.index(closest_edge)
                if self.gui:
                    self.gui.add_task_particle(self.location[:], closest_edge.location, particle_color, task.id)
                    self.gui.stats['tasks_offloaded'] += 1
                    stat_key = f'edge_{edge_id}'
                    self.gui.stats[stat_key] = self.gui.stats.get(stat_key, 0) + 1
                
                # Transmission & Computation Energy
                tx_energy = TRANSMISSION_POWER * transmission_time * ENERGY_SCALE_FACTOR
                comp_energy_overhead = 0.05 * local_comp_energy_pred # Basic control overhead
                self.battery -= (tx_energy + comp_energy_overhead)
                
                yield self.env.timeout(transmission_time)
                closest_edge.current_load += 1
                with closest_edge.resource.request() as req:
                    yield req
                    yield self.env.process(closest_edge.process_task(task))
            
            else: # LOCAL
                self.current_target = None
                self.battery -= local_comp_energy_pred
                processing_time = task.cpu_cycles / DEFAULT_CPU_FREQ
                yield self.env.timeout(processing_time)
                task.completion_time = self.env.now
                print(f"  -> Task {task.id} processed LOCALLY in {processing_time:.4f}s")

            total_delay = task.completion_time - task.creation_time
            print(f"  -> Task {task.id} COMPLETED. Total Delay: {total_delay:.4f}s | Battery: {self.battery:.1f}J")
            
            # Reset GUI state
            self.current_target = None


def main(): # Renamed from run_simulation
    global LLM_ANALYZER
    
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

if __name__ == "__main__":
    main()
