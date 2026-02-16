import simpy
import random
import math
import dataclasses
from enum import Enum
from llm_analyzer import SemanticAnalyzer
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

# Global LLM Analyzer (initialized once)
LLM_ANALYZER = None

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
            
            # --- OFFLOADING DECISION (Placeholder for RL/LLM Agent) ---
            # Ideally this comes from the Intelligence Layer
            # For now: Greedy Strategy (Offload to closest Edge if Signal is good, else Local)
            
            closest_edge = min(self.edge_servers, key=lambda e: math.dist(self.location, e.location))
            datarate, distance = self.channel.calculate_datarate(self, closest_edge)
            
            transmission_time = task.size_bits / datarate
            # Simple Greedy Logic: 
            # If Edge is close/fast, go Edge.
            # If Task is HUGE, go Cloud.
            
            if task.size_bits > 4e6: # > 4Mb -> Cloud
                print(f"  -> Decided to OFFLOAD to CLOUD (Heavy Task)")
                self.current_target = self.cloud_server
                
                # Trigger GUI particle animation
                if self.gui:
                    self.gui.add_task_particle(self.location[:], (900, 100), (50, 50, 200))
                    self.gui.stats['tasks_offloaded'] += 1
                    self.gui.stats['tasks_to_cloud'] += 1
                
                # Cloud Latency
                yield self.env.process(self.cloud_server.process_task(task))
            else:
                print(f"  -> Decided to OFFLOAD to Edge-{closest_edge.id} (Dist: {distance:.2f}m, Rate: {datarate/1e6:.2f} Mbps)")
                self.current_target = closest_edge
                
                # Trigger GUI particle animation
                if self.gui:
                    self.gui.add_task_particle(self.location[:], closest_edge.location, (255, 165, 0))
                    self.gui.stats['tasks_offloaded'] += 1
                    self.gui.stats['tasks_to_edge'] += 1
                
                # Transmission Cost (Energy)
                tx_energy = TRANSMISSION_POWER * transmission_time * ENERGY_SCALE_FACTOR
                
                # Computation Energy (if processing locally after receiving from edge)
                # This represents the control overhead
                comp_energy = KAPPA * (DEFAULT_CPU_FREQ ** 2) * DEFAULT_CPU_FREQ * (task.cpu_cycles / DEFAULT_CPU_FREQ) * ENERGY_SCALE_FACTOR
                
                # Total energy consumption
                total_energy = tx_energy + comp_energy
                self.battery -= total_energy
                
                print(f"  -> Energy: TX={tx_energy:.2f}J, Comp={comp_energy:.2f}J, Battery: {self.battery:.1f}J ({(self.battery/BATTERY_CAPACITY)*100:.1f}%)")
                
                yield self.env.timeout(transmission_time) # Transmission delay
                
                # Request processing at Edge
                closest_edge.current_load += 1
                with closest_edge.resource.request() as req:
                    yield req
                    yield self.env.process(closest_edge.process_task(task))

            total_delay = task.completion_time - task.creation_time
            print(f"  -> Task {task.id} COMPLETED. Total Delay: {total_delay:.4f}s")
            
            # Reset GUI state
            self.current_target = None


def main(): # Renamed from run_simulation
    global LLM_ANALYZER
    
    random.seed(RANDOM_SEED)
    env = simpy.Environment()
    
    # Initialize LLM Semantic Analyzer (rule-based by default)
    print("[INIT] Initializing LLM Semantic Analyzer...")
    LLM_ANALYZER = SemanticAnalyzer(use_llm=False)  # Set to True for actual LLM
    print("[INIT] Analyzer ready!")
    
    # Infrastructure
    channel = WirelessChannel()
    cloud_server = CloudServer(env)
    edge_servers = [
        EdgeServer(env, id=0, location=(250, 250), max_freq=2e9),   # 2 GHz
        EdgeServer(env, id=1, location=(750, 250), max_freq=2.5e9), # 2.5 GHz
        EdgeServer(env, id=2, location=(500, 750), max_freq=3e9)    # 3 GHz
    ]
    
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
