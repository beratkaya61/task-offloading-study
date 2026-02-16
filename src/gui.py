import pygame
import math

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 50, 50)     # Critical Task / Low Battery
GREEN = (50, 200, 50)   # Edge Server
BLUE = (50, 50, 200)    # Cloud
GRAY = (100, 100, 100)  # Device
YELLOW = (200, 200, 50) # Interference / Wait
ORANGE = (255, 165, 0)  # High Data Task
LIGHT_GRAY = (240, 240, 240)  # Background grid

SCREEN_WIDTH = 1200  # Increased for legend panel
SCREEN_HEIGHT = 1000

class TaskParticle:
    """Visual representation of a task being offloaded"""
    def __init__(self, start_pos, end_pos, color):
        self.pos = list(start_pos)
        self.target = end_pos
        self.color = color
        self.speed = 5
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
    
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.pos[0]), int(self.pos[1])), 4)

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
        
        self.particles = []  # Task flow animations
        self.stats = {"tasks_offloaded": 0, "tasks_to_cloud": 0, "tasks_to_edge": 0}
        
        self.running = True

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                exit()

    def add_task_particle(self, start, end, color):
        """Add animated particle for task flow"""
        self.particles.append(TaskParticle(start, end, color))

    def draw_grid(self):
        """Draw background grid"""
        for x in range(0, 1000, 100):
            pygame.draw.line(self.screen, LIGHT_GRAY, (x, 0), (x, 1000), 1)
        for y in range(0, 1000, 100):
            pygame.draw.line(self.screen, LIGHT_GRAY, (0, y), (1000, y), 1)

    def draw_legend(self):
        """Draw legend panel on the right side"""
        panel_x = 1020
        panel_width = 180
        
        # Background
        pygame.draw.rect(self.screen, (250, 250, 250), (panel_x, 0, panel_width, SCREEN_HEIGHT))
        pygame.draw.line(self.screen, BLACK, (panel_x, 0), (panel_x, SCREEN_HEIGHT), 2)
        
        # Title
        title = self.title_font.render("LEGEND", True, BLACK)
        self.screen.blit(title, (panel_x + 60, 10))
        
        y_offset = 50
        
        # Icons
        legends = [
            ("🚗", "Mobile IoT (Vehicles)", BLACK),
            ("🏢", "Edge Server", GREEN),
            ("☁️", "Cloud Server", BLUE),
        ]
        
        for icon, label, color in legends:
            icon_surf = self.icon_font.render(icon, True, color)
            self.screen.blit(icon_surf, (panel_x + 10, y_offset - 5))
            text = self.font.render(label, True, BLACK)
            self.screen.blit(text, (panel_x + 50, y_offset + 5))
            y_offset += 45
        
        # Stats Panel
        y_offset += 30
        stats_title = self.title_font.render("STATISTICS", True, BLACK)
        self.screen.blit(stats_title, (panel_x + 40, y_offset))
        y_offset += 30
        
        stats_texts = [
            f"Time: {self.env.now:.1f}s",
            f"Total Tasks: {self.stats['tasks_offloaded']}",
            f"→ Edge: {self.stats['tasks_to_edge']}",
            f"→ Cloud: {self.stats['tasks_to_cloud']}",
            f"Active Devices: {len([d for d in self.devices if d.battery > 0])}",
        ]
        
        for stat in stats_texts:
            text = self.font.render(stat, True, BLACK)
            self.screen.blit(text, (panel_x + 10, y_offset))
            y_offset += 20

    def draw(self):
        if not self.running: return
        
        self.screen.fill(WHITE)
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
                particle.draw(self.screen)
            else:
                self.particles.remove(particle)
        
        # Draw legend panel
        self.draw_legend()
        
        pygame.display.flip()
        self.handle_events()
        
    def update(self):
        self.draw()
        self.clock.tick(30)  # 30 FPS
