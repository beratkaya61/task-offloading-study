import pygame
import math

# Colors - Modern Professional Palette
WHITE = (255, 255, 255)
BLACK = (20, 20, 25)      # Deep Black (Background)
NAVY = (10, 25, 45)       # Dark Navy (Glassmorphism base)
ACID_GREEN = (175, 255, 45) # Sharp Green for accents
CYAN = (0, 255, 255)      # Cyber Cyan
RED = (255, 80, 80)       # Modern Red
GREEN = (80, 255, 150)    # Soft Neon Green
BLUE = (80, 180, 255)     # Modern Blue
GRAY = (180, 180, 190)    # Text Gray
DARK_GRAY = (40, 40, 50)  # Panel Gray
ORANGE = (255, 160, 0)
GOLD = (255, 215, 0)
LIGHT_GRAY = (240, 240, 250)

SCREEN_WIDTH = 1500  # Map (1000) + Side Panel (500)
SCREEN_HEIGHT = 1000
SIDE_PANEL_X = 1000
SIDE_PANEL_WIDTH = 500

class TaskParticle:
    """Visual representation of a task being offloaded"""
    def __init__(self, start_pos, end_pos, color, task_id):
        self.pos = list(start_pos)
        self.target = end_pos
        self.color = color
        self.task_id = task_id
        self.speed = 7 # Slightly faster for better flow
        self.alive = True
        
    def update(self):
        # Move towards target
        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist < self.speed:
            self.alive = False
            return
        
        self.pos[0] += (dx / dist) * self.speed
        self.pos[1] += (dy / dist) * self.speed
    
    def draw(self, screen, font):
        pygame.draw.circle(screen, self.color, (int(self.pos[0]), int(self.pos[1])), 5)
        # Draw Task ID pill
        id_text = font.render(f"T-{self.task_id}", True, WHITE)
        text_rect = id_text.get_rect(center=(int(self.pos[0]), int(self.pos[1]) - 15))
        pygame.draw.rect(screen, self.color, text_rect.inflate(4, 2), border_radius=3)
        screen.blit(id_text, text_rect)

class SimulationGUI:
    def __init__(self, env, devices, edge_servers, cloud_server):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("IoT Task Offloading Simulation (Device -> Edge -> Cloud)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 12)
        self.title_font = pygame.font.SysFont("Arial", 16, bold=True)
        
        # Windows supports 'segoeuiemoji' for icons, fallback to Arial
        try:
            self.icon_font = pygame.font.SysFont("segoeuiemoji", 40)  # Larger icons
            self.small_icon_font = pygame.font.SysFont("segoeuiemoji", 20)
        except:
            self.icon_font = pygame.font.SysFont("Arial", 40)
            self.small_icon_font = pygame.font.SysFont("Arial", 20)
        
        self.env = env
        self.devices = devices
        self.edge_servers = edge_servers
        self.cloud_server = cloud_server
        
        # UI Setup
        self.particles = []  # Task flow animations
        self.stats = {"tasks_offloaded": 0, "tasks_to_cloud": 0, "tasks_to_edge": 0}
        self.decision_log = [] # List of strings for the chat panel
        self.max_log_msgs = 12 # Reduced for better spacing
        
        # Transparent Surfaces for Glassmorphism
        self.side_panel_surf = pygame.Surface((SIDE_PANEL_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        self.side_panel_surf.fill((10, 25, 45, 240)) # Arka plan (Navy with alpha)
        
        self.log_panel_surf = pygame.Surface((SIDE_PANEL_WIDTH - 20, 480), pygame.SRCALPHA)
        self.log_panel_surf.fill((20, 20, 30, 180)) # Decision log bg
        
        self.running = True

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                exit()

    def add_task_particle(self, start, end, color, task_id):
        """Add animated particle for task flow with ID"""
        self.particles.append(TaskParticle(start, end, color, task_id))

    def add_decision_log(self, task_id, message, color=BLACK):
        """Add a detailed decision reasoning to the chat log"""
        self.decision_log.append({"id": task_id, "msg": message, "color": color, "time": self.env.now})
        if len(self.decision_log) > self.max_log_msgs:
            self.decision_log.pop(0)

    def draw_grid(self):
        """Draw background grid"""
        for x in range(0, 1000, 100):
            pygame.draw.line(self.screen, LIGHT_GRAY, (x, 0), (x, 1000), 1)
        for y in range(0, 1000, 100):
            pygame.draw.line(self.screen, LIGHT_GRAY, (0, y), (1000, y), 1)

    def draw_legend(self):
        """Draw professional legend and performance metrics"""
        y_offset = 20
        
        # --- Dashboard Header ---
        header_text = self.title_font.render("IOT DASHBOARD", True, CYAN)
        self.screen.blit(header_text, (SIDE_PANEL_X + 25, y_offset))
        y_offset += 35
        
        # --- Infrastructure Card ---
        infra_rect = pygame.Rect(SIDE_PANEL_X + 20, y_offset, SIDE_PANEL_WIDTH - 40, 150)
        pygame.draw.rect(self.screen, (30, 35, 60), infra_rect, border_radius=8)
        pygame.draw.rect(self.screen, CYAN, infra_rect, 1, border_radius=8)
        
        sub_title = self.font.render("INFRASTRUCTURE STATUS", True, CYAN)
        self.screen.blit(sub_title, (SIDE_PANEL_X + 35, y_offset + 10))
        y_offset += 40
        
        # Changed icons to standard symbols for better compatibility
        legends = [
            ("[D]", "V-IoT Device", WHITE, (100, 100, 100)),
            ("[E]", "Edge Compute Node", GREEN, (50, 200, 50)),
            ("[C]", "Remote Cloud", BLUE, (80, 80, 255)),
        ]
        
        for icon, label, color, dot_color in legends:
            # Draw a colored dot and text instead of emojis
            pygame.draw.circle(self.screen, dot_color, (SIDE_PANEL_X + 45, y_offset + 10), 8)
            icon_surf = self.font.render(icon, True, color)
            self.screen.blit(icon_surf, (SIDE_PANEL_X + 65, y_offset + 2))
            text = self.font.render(label, True, WHITE)
            self.screen.blit(text, (SIDE_PANEL_X + 110, y_offset + 3))
            y_offset += 30

        # --- Performance Metrics Card ---
        y_offset += 25
        stats_rect = pygame.Rect(SIDE_PANEL_X + 20, y_offset, SIDE_PANEL_WIDTH - 40, 130)
        pygame.draw.rect(self.screen, (25, 30, 50), stats_rect, border_radius=8)
        
        stats_list = [
            ("Uptime:", f"{self.env.now:.1f}s", ACID_GREEN),
            ("Offloaded:", f"{self.stats['tasks_offloaded']}", WHITE),
            ("Edge Load:", f"{self.stats['tasks_to_edge']}", GREEN),
            ("Cloud Load:", f"{self.stats['tasks_to_cloud']}", BLUE),
            ("Devices Online:", f"{len([d for d in self.devices if d.battery > 0])}", ACID_GREEN),
        ]
        
        for label, val, color in stats_list:
            lbl_surf = self.font.render(label, True, GRAY)
            val_surf = self.font.render(val, True, color)
            self.screen.blit(lbl_surf, (SIDE_PANEL_X + 35, y_offset + 10))
            self.screen.blit(val_surf, (SIDE_PANEL_X + 220, y_offset + 10))
            y_offset += 20

    def draw_knowledge_base(self):
        """Separate panel for Scientific Booklet to avoid overlapping"""
        panel_y = 360
        panel_h = 160
        booklet_rect = pygame.Rect(SIDE_PANEL_X + 20, panel_y, SIDE_PANEL_WIDTH - 40, panel_h)
        pygame.draw.rect(self.screen, (35, 40, 65), booklet_rect, border_radius=8)
        pygame.draw.rect(self.screen, GOLD, booklet_rect, 1, border_radius=8)
        
        title = self.font.render("SCIENTIFIC BOOKLET", True, GOLD)
        self.screen.blit(title, (SIDE_PANEL_X + 35, panel_y + 10))
        
        lines = [
            "1. Shannon: R = B * log2(1 + SNR)",
            "2. Energy: P_cpu = k * f^3",
            "3. SNR: 10 * log10(P_rx / N)",
            "4. Task: Real-world Google Trace",
            "5. Logic: Semantic AI Offloading"
        ]
        for i, line in enumerate(lines):
            line_surf = self.font.render(line, True, WHITE)
            self.screen.blit(line_surf, (SIDE_PANEL_X + 35, panel_y + 40 + (i * 22)))

    def draw_decision_log(self):
        """Draw Professional AI Decision Feed in its own space"""
        panel_y = 535
        panel_h = 445
        
        # Log background with glass effect
        self.screen.blit(self.log_panel_surf, (SIDE_PANEL_X + 10, panel_y))
        pygame.draw.rect(self.screen, CYAN, (SIDE_PANEL_X + 10, panel_y, SIDE_PANEL_WIDTH - 20, panel_h), 1, border_radius=5)
        
        title_surf = self.title_font.render("SEMANTIC DECISION FEED", True, ACID_GREEN)
        self.screen.blit(title_surf, (SIDE_PANEL_X + 25, panel_y + 15))
        
        y_pos = panel_y + 50
        for entry in self.decision_log[::-1]:
            # Container for each log entry
            entry_h = 95 # Increased height for more detail
            if y_pos + entry_h > panel_y + panel_h: break
            
            # Entry border/indicator
            pygame.draw.line(self.screen, entry['color'], (SIDE_PANEL_X + 20, y_pos), (SIDE_PANEL_X + 20, y_pos + entry_h - 15), 3)
            
            # Time & ID
            header = f"Task-{entry['id']} | Mission T: {entry['time']:.1f}s"
            header_surf = self.font.render(header, True, entry['color'])
            self.screen.blit(header_surf, (SIDE_PANEL_X + 30, y_pos + 5))
            
            # Message lines
            messages = entry['msg'].split('\n')
            for i, line in enumerate(messages):
                # Bold the 'Karar' or 'Decision' line
                c = WHITE if i > 0 else entry['color']
                if "Karar:" in line or "Decision:" in line: c = ACID_GREEN
                
                line_surf = self.font.render(line, True, c)
                self.screen.blit(line_surf, (SIDE_PANEL_X + 35, y_pos + 25 + (i * 13)))
            
            y_pos += entry_h

    def draw(self):
        if not self.running: return
        
        self.screen.fill((25, 25, 30)) # Modern Dark background
        self.draw_grid()
        
        # Draw Cloud (Top Right)
        cloud_icon = self.icon_font.render("☁️", True, BLUE)
        self.screen.blit(cloud_icon, (880, 80))
        text = self.font.render("CLOUD", True, BLACK)
        self.screen.blit(text, (890, 120))
        
        # Cloud queue indicator
        cloud_queue_ratio = min(1.0, self.cloud_server.queue_length / self.cloud_server.max_queue_size)
        cloud_queue_width = 60
        cloud_queue_height = 10
        cloud_queue_x, cloud_queue_y = 870, 140
        
        pygame.draw.rect(self.screen, LIGHT_GRAY, (cloud_queue_x, cloud_queue_y, cloud_queue_width, cloud_queue_height))
        if cloud_queue_ratio > 0.7:
            cloud_queue_color = RED
        elif cloud_queue_ratio > 0.4:
            cloud_queue_color = ORANGE
        else:
            cloud_queue_color = BLUE
        pygame.draw.rect(self.screen, cloud_queue_color, (cloud_queue_x, cloud_queue_y, int(cloud_queue_width * cloud_queue_ratio), cloud_queue_height))
        pygame.draw.rect(self.screen, BLACK, (cloud_queue_x, cloud_queue_y, cloud_queue_width, cloud_queue_height), 1)
        
        cloud_queue_text = self.font.render(f"Queue: {self.cloud_server.queue_length} | Load: {self.cloud_server.current_load}", True, BLACK)
        self.screen.blit(cloud_queue_text, (870, 153))
        
        # Draw Edge Servers
        for edge in self.edge_servers:
            x, y = edge.location
            icon = self.icon_font.render("🏢", True, GREEN)
            rect = icon.get_rect(center=(x, y))
            self.screen.blit(icon, rect)
            
            # Load indicator
            load_text = self.font.render(f"Load: {edge.current_load}", True, BLACK)
            self.screen.blit(load_text, (x-25, y+20))
            
            # Queue length bar (read directly from SimPy Resource)
            queue_length = len(edge.resource.queue)
            queue_ratio = min(1.0, queue_length / edge.max_queue_size)
            queue_bar_width = 50
            queue_bar_height = 8
            queue_x, queue_y = x - 25, y + 35
            
            # Background
            pygame.draw.rect(self.screen, LIGHT_GRAY, (queue_x, queue_y, queue_bar_width, queue_bar_height))
            # Filled (red if queue is full, yellow if medium, green if low)
            if queue_ratio > 0.7:
                queue_color = RED
            elif queue_ratio > 0.4:
                queue_color = ORANGE
            else:
                queue_color = GREEN
            pygame.draw.rect(self.screen, queue_color, (queue_x, queue_y, int(queue_bar_width * queue_ratio), queue_bar_height))
            pygame.draw.rect(self.screen, BLACK, (queue_x, queue_y, queue_bar_width, queue_bar_height), 1)
            
            # Queue text
            queue_text = self.font.render(f"Q: {queue_length}", True, BLACK)
            self.screen.blit(queue_text, (x - 15, y + 45))
            
            # CPU frequency bar
            freq_ratio = edge.current_freq / edge.max_freq
            bar_width = 40
            bar_height = 5
            pygame.draw.rect(self.screen, GRAY, (x-20, y+58, bar_width, bar_height))
            pygame.draw.rect(self.screen, GREEN, (x-20, y+58, int(bar_width * freq_ratio), bar_height))
            
        # Draw Devices
        for device in self.devices:
            x, y = int(device.location[0]), int(device.location[1])
            
            # Battery status color (normalize to percentage)
            battery_pct = (device.battery / 10000.0) * 100  # Normalize Joules to percentage
            battery_pct = max(0, min(100, battery_pct))  # Clamp to 0-100
            
            if battery_pct < 20:
                status_color = RED
                status_icon = "⚠️"
            elif battery_pct < 50:
                status_color = ORANGE
                status_icon = "⚡"
            else:
                status_color = GREEN
                status_icon = "✓"
            
            # Draw larger device icon with background circle
            bg_radius = 25
            pygame.draw.circle(self.screen, (240, 240, 240), (x, y), bg_radius)  # Background
            pygame.draw.circle(self.screen, status_color, (x, y), bg_radius, 3)  # Status ring
            
            # Device icon (Mobile: car, Static: sensor)
            device_icon = self.icon_font.render("🚗", True, BLACK)
            rect = device_icon.get_rect(center=(x, y))
            self.screen.blit(device_icon, rect)
            
            # Battery percentage text (large and visible)
            battery_text = self.font.render(f"{int(battery_pct)}%", True, status_color)
            self.screen.blit(battery_text, (x - 15, y + 30))
            
            # Battery bar with gradient effect
            bar_x, bar_y = x - 20, y - 35
            bar_width, bar_height = 40, 6
            
            # Background
            pygame.draw.rect(self.screen, GRAY, (bar_x, bar_y, bar_width, bar_height))
            
            # Filled portion (color based on battery level)
            fill_width = int(bar_width * (battery_pct / 100.0))
            pygame.draw.rect(self.screen, status_color, (bar_x, bar_y, fill_width, bar_height))
            
            # Border
            pygame.draw.rect(self.screen, BLACK, (bar_x, bar_y, bar_width, bar_height), 1)
            
            # Status icon (small, top left)
            status_surf = self.small_icon_font.render(status_icon, True, status_color)
            self.screen.blit(status_surf, (x - 30, y - 30))
            
            # Draw Active Links (Lines)
            if device.current_target:
                if hasattr(device.current_target, 'location'):
                    target_loc = device.current_target.location
                else:
                    # Cloud
                    target_loc = (900, 100)
                
                line_color = ORANGE if device.current_task_type == "HIGH_DATA" else BLUE
                pygame.draw.line(self.screen, line_color, (x, y), target_loc, 2)

        # Update and draw particles
        for particle in self.particles[:]:
            particle.update()
            if particle.alive:
                particle.draw(self.screen, self.font)
            else:
                self.particles.remove(particle)
        
        # Draw Side Panel Background (Glassmorphism Effect)
        self.screen.blit(self.side_panel_surf, (SIDE_PANEL_X, 0))
        pygame.draw.line(self.screen, CYAN, (SIDE_PANEL_X, 0), (SIDE_PANEL_X, SCREEN_HEIGHT), 2)
        
        # Draw panels
        self.draw_legend()
        self.draw_knowledge_base()
        self.draw_decision_log()
        
        pygame.display.flip()
        self.handle_events()
        
    def update(self):
        self.draw()
        self.clock.tick(30)  # 30 FPS
