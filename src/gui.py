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
PURPLE = (200, 100, 255)

SCREEN_WIDTH = 1800  # Methodology (400) + Map (1000) + Side Panel (400)
SCREEN_HEIGHT = 1000

METHOD_PANEL_X = 0
METHOD_PANEL_WIDTH = 400

MAP_X = 400
MAP_WIDTH = 1000

SIDE_PANEL_X = 1400
SIDE_PANEL_WIDTH = 400

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
    
    def draw(self, screen, font, zoom_level, offset):
        sx = MAP_X + (self.pos[0] * zoom_level) + offset[0]
        sy = (self.pos[1] * zoom_level) + offset[1]
        
        # Only draw if within map bounds
        if MAP_X <= sx <= SIDE_PANEL_X:
            pygame.draw.circle(screen, self.color, (int(sx), int(sy)), int(5 * zoom_level))
            # Draw Task ID pill (scaled)
            id_text = font.render(f"T-{self.task_id}", True, WHITE)
            text_rect = id_text.get_rect(center=(int(sx), int(sy) - 15))
            pygame.draw.rect(screen, self.color, text_rect.inflate(4, 2), border_radius=3)
            screen.blit(id_text, text_rect)

class SimulationGUI:
    def __init__(self, env, devices, edge_servers, cloud_server):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("IoT Task Offloading Simulation (Device -> Edge -> Cloud)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 14) # Increased base font
        self.small_font = pygame.font.SysFont("Arial", 12)
        self.title_font = pygame.font.SysFont("Arial", 18, bold=True) # Increased title font
        
        # Robust Unicode Symbols (Fallbacks for Windows/Unusual Fonts)
        # 📡 = (U+1F4E1), ⚡ = (U+26A1), 🧠 = (U+1F9E0), 🤖 = (U+1F916), 🎓 = (U+1F393), 🏆 = (U+1F3C6)
        # If emoji font fails, we use descriptive labels
        try:
            self.icon_font = pygame.font.SysFont("segoeuiemoji", 32)
            self.small_icon_font = pygame.font.SysFont("segoeuiemoji", 18)
        except:
            self.icon_font = pygame.font.SysFont("Arial", 32)
            self.small_icon_font = pygame.font.SysFont("Arial", 18)
        
        self.env = env
        self.devices = devices
        self.edge_servers = edge_servers
        self.cloud_server = cloud_server
        
        # UI Setup
        self.particles = []  # Task flow animations
        self.stats = {
            "tasks_offloaded": 0, "tasks_to_cloud": 0, "tasks_to_edge": 0,
            "ppo_lat": 0, "ppo_en": 0,
            "random_lat": 0, "random_en": 0,
            "greedy_lat": 0, "greedy_en": 0,
            "fairness_index": 1.0, "jitter_avg": 0.0, "qoe_score": 100.0
        }
        self.decision_log = [] # List of strings for the chat panel
        
        # Transparent Surfaces for Glassmorphism
        self.method_panel_surf = pygame.Surface((METHOD_PANEL_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        self.method_panel_surf.fill((15, 20, 35, 250)) # Distinct Darker Navy
        
        self.side_panel_surf = pygame.Surface((SIDE_PANEL_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        self.side_panel_surf.fill((10, 25, 45, 240)) # Arka plan (Navy with alpha)
        
        self.log_panel_surf = pygame.Surface((SIDE_PANEL_WIDTH - 20, 500), pygame.SRCALPHA)
        self.log_panel_surf.fill((20, 20, 30, 180)) # Decision log bg
        
        # Zoom & Pan State
        self.zoom_level = 1.0
        self.map_offset = [0, 0]
        self.min_zoom = 0.5
        self.max_zoom = 3.0
        
        # Scrolling State for Feed
        self.scroll_y = 0
        self.max_scroll = 0
        
        # Toast Notification State
        self.toast_msg = ""
        self.toast_timer = 0
        self.toast_alpha = 0
        
        self.running = True

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                exit()
            
            # Zoom and Scroll Handling
            if event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if mx > SIDE_PANEL_X: # Scroll the log feed
                    self.scroll_y += event.y * 30
                    if self.scroll_y > 0: self.scroll_y = 0
                    if self.scroll_y < self.max_scroll: self.scroll_y = self.max_scroll
                elif mx > METHOD_PANEL_WIDTH and mx < SIDE_PANEL_X: # Zoom the map
                    old_zoom = self.zoom_level
                    self.zoom_level += event.y * 0.1
                    self.zoom_level = max(self.min_zoom, min(self.max_zoom, self.zoom_level))
                    
                    # Zoom towards cursor
                    zoom_factor = self.zoom_level / old_zoom
                    self.map_offset[0] = (self.map_offset[0] - (mx - MAP_X)) * zoom_factor + (mx - MAP_X)
                    self.map_offset[1] = (self.map_offset[1] - my) * zoom_factor + my

    def add_task_particle(self, start, end, color, task_id):
        """Add animated particle for task flow with ID"""
        self.particles.append(TaskParticle(start, end, color, task_id))

    def add_decision_log(self, task_id, message, color=BLACK):
        """Add a detailed decision reasoning to the chat log"""
        self.decision_log.append({"id": task_id, "msg": message, "color": color, "time": self.env.now})
        # Removed hard pop to allow scrolling through history
        if len(self.decision_log) > 100: # Keep a reasonable buffer
            self.decision_log.pop(0)

    def draw_grid(self):
        """Draw background grid with zoom transformations"""
        step = int(100 * self.zoom_level)
        if step < 10: step = 10
        
        for x in range(0, 1000, 100):
            sx = MAP_X + (x * self.zoom_level) + self.map_offset[0]
            if MAP_X <= sx <= SIDE_PANEL_X:
                pygame.draw.line(self.screen, (40, 45, 60), (sx, 0), (sx, 1000), 1)
                
        for y in range(0, 1000, 100):
            sy = (y * self.zoom_level) + self.map_offset[1]
            pygame.draw.line(self.screen, (40, 45, 60), (MAP_X, sy), (SIDE_PANEL_X, sy), 1)

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
        # Total Metrics
        stats_list = [
            ("Uptime:", f"{self.env.now:.1f}s", ACID_GREEN),
            ("Total Tasks:", f"{self.stats['tasks_offloaded']}", WHITE),
            ("Devices Online:", f"{len([d for d in self.devices if d.battery > 0])}", ACID_GREEN),
        ]
        
        for label, val, color in stats_list:
            lbl_surf = self.font.render(label, True, GRAY)
            val_surf = self.font.render(val, True, color)
            self.screen.blit(lbl_surf, (SIDE_PANEL_X + 35, y_offset + 10))
            self.screen.blit(val_surf, (SIDE_PANEL_X + 180, y_offset + 10))
            y_offset += 20

        # Detailed Breakdown - Reduced vertical footprint to avoid overlap
        y_offset += 10
        breakdown_title = self.font.render("OFFLOAD DISTRIBUTION", True, CYAN)
        self.screen.blit(breakdown_title, (SIDE_PANEL_X + 35, y_offset))
        y_offset += 20
        
        cloud_count = self.stats.get('tasks_to_cloud', 0)
        c_surf = self.small_font.render(f"➤ Cloud Gateway: {cloud_count}", True, BLUE)
        self.screen.blit(c_surf, (SIDE_PANEL_X + 45, y_offset))
        y_offset += 16
        
        for i, edge in enumerate(self.edge_servers):
            count = self.stats.get(f'edge_{i}', 0)
            e_surf = self.small_font.render(f"➤ Edge Node-{i}: {count}", True, GREEN)
            self.screen.blit(e_surf, (SIDE_PANEL_X + 45, y_offset))
            y_offset += 16

    def draw_methodology_panel(self):
        """Enhanced Left Panel with detailed Turkish scientific explanations"""
        self.screen.blit(self.method_panel_surf, (0, 0))
        pygame.draw.line(self.screen, GOLD, (METHOD_PANEL_WIDTH, 0), (METHOD_PANEL_WIDTH, SCREEN_HEIGHT), 2)
        
        y = 30
        # Robust Icon Header: Golden dot + Text
        pygame.draw.circle(self.screen, GOLD, (20, y + 10), 6)
        title = self.title_font.render("SİMÜLASYON METODOLOJİSİ", True, GOLD)
        self.screen.blit(title, (35, y))
        y += 50
        
        sections = [
            ("KABLOSUZ AĞ (Shannon)", CYAN, [
                "Yöntem: Shannon-Hartley Kapasite modeli.",
                "Bilimsel Veri: SNR ve mesafe bazlı anlık",
                "bant genişliği dalgalanmalarını simüle eder."
            ]),
            ("ENERJİ (DVFS Modeli)", CYAN, [
                "Yöntem: Dinamik Voltaj ve Frekans Ölçekleme.",
                "Bilimsel Veri: Kararın batarya ömrüne etkisi",
                "P=k*f³ formülü ile saniyelik hesaplanır."
            ]),
            ("SEMANTİK YAPAY ZEKA (LLM)", ACID_GREEN, [
                "Yöntem: NLP bazlı görev önceliklendirme.",
                "Bilimsel Veri: Kritiklik (Hastalık vs. Log)",
                "context analizi ile dinamik puan üretir."
            ])
        ]
        
        sub_header_font = pygame.font.SysFont("Arial", 15, bold=True)
        text_font = pygame.font.SysFont("Arial", 13)
        
        for title, color, bullets in sections:
            # Section Indicator: Colored square
            pygame.draw.rect(self.screen, color, (20, y + 5, 10, 10))
            t_surf = sub_header_font.render(title, True, color)
            self.screen.blit(t_surf, (40, y))
            y += 22
            for line in bullets:
                l_surf = text_font.render(line, True, LIGHT_GRAY)
                self.screen.blit(l_surf, (40, y))
                y += 18
            y += 25

        # --- AI KARŞILAŞTIRMA BÖLÜMÜ (Turkish Deep-Dive) ---
        y += 10
        pygame.draw.circle(self.screen, GOLD, (25, y+10), 5)
        comp_title = sub_header_font.render("ALGORİTMA KIYASLAMASI", True, GOLD)
        self.screen.blit(comp_title, (40, y))
        y += 25
        
        # Why we prefer PPO?
        ppo_desc = [
            "PPO (Yapay Zeka): Neden Tercih Ediyoruz?",
            "• Öngörülü (Proactive): Kuyruk yoğunluğunu",
            "  ve ağ dar boğazlarını önceden tahmin eder.",
            "• Global Optimizasyon: Sadece anı değil,",
            "  uzun vadeli sistem ödülünü maksimize eder."
        ]
        
        for i, line in enumerate(ppo_desc):
            color = ACID_GREEN if i == 0 else WHITE
            l_surf = self.small_font.render(line, True, color)
            self.screen.blit(l_surf, (30, y))
            y += 18
        y += 15
        
        others = [
            ("Greedy (Açgözlü):", "Anlık en hızlıyı seçer, dar görüşlüdür."),
            ("Random (Baz):", "Literatürdeki alt kıstas noktasıdır.")
        ]
        for name, desc in others:
            n_surf = text_font.render(name, True, ORANGE)
            self.screen.blit(n_surf, (30, y))
            d_surf = self.small_font.render(desc, True, GRAY)
            self.screen.blit(d_surf, (35, y + 16))
            y += 35
            
    def draw_knowledge_base(self):
        """Redesigned panel for Node-specific Health Status - Pushed down to avoid overlap"""
        panel_y = 380 # Pushed from 360
        panel_h = 145 # Slimmed down
        booklet_rect = pygame.Rect(SIDE_PANEL_X + 20, panel_y, SIDE_PANEL_WIDTH - 40, panel_h)
        pygame.draw.rect(self.screen, (35, 40, 65), booklet_rect, border_radius=8)
        pygame.draw.rect(self.screen, GOLD, booklet_rect, 1, border_radius=8)
        
        y = panel_y + 15
        title = self.font.render("NODE HEALTH STATUS", True, GOLD)
        self.screen.blit(title, (SIDE_PANEL_X + 35, y))
        y += 25
        
        # Cloud Health
        cloud_color = BLUE if self.cloud_server.queue_length < 15 else RED
        pygame.draw.circle(self.screen, cloud_color, (SIDE_PANEL_X + 45, y + 8), 6)
        c_status = "OVERLOAD" if cloud_color == RED else "STABLE"
        c_health = self.font.render(f"Cloud Server: {c_status}", True, WHITE)
        self.screen.blit(c_health, (SIDE_PANEL_X + 65, y))
        y += 22
        
        # Individual Edge Health
        for i, edge in enumerate(self.edge_servers):
            q = len(edge.resource.queue)
            e_color = GREEN if q < 4 else (ORANGE if q < 8 else RED)
            pygame.draw.circle(self.screen, e_color, (SIDE_PANEL_X + 45, y + 8), 6)
            e_status = "STABLE" if q < 4 else ("BUSY" if q < 8 else "HEAVY")
            e_health = self.font.render(f"Edge Node {i}: {e_status} (Q:{q})", True, WHITE)
            self.screen.blit(e_health, (SIDE_PANEL_X + 65, y))
            y += 20

    def draw_decision_log(self):
        """Draw Professional AI Decision Feed in its own space with Scrolling"""
        panel_y = 535
        panel_h = 445
        
        # Log background with glass effect
        self.screen.blit(self.log_panel_surf, (SIDE_PANEL_X + 10, panel_y))
        pygame.draw.rect(self.screen, CYAN, (SIDE_PANEL_X + 10, panel_y, SIDE_PANEL_WIDTH - 20, panel_h), 1, border_radius=5)
        
        title_surf = self.title_font.render("SEMANTIC DECISION FEED", True, ACID_GREEN)
        self.screen.blit(title_surf, (SIDE_PANEL_X + 25, panel_y + 15))
        
        # Create a clipping surface for the log entries
        clip_rect = pygame.Rect(SIDE_PANEL_X + 15, panel_y + 50, SIDE_PANEL_WIDTH - 30, panel_h - 60)
        log_surface = pygame.Surface((clip_rect.width, 5000), pygame.SRCALPHA) # Large enough for many logs
        
        y_cursor = 0
        entry_h = 100
        
        # Draw from newest to oldest in the log_surface
        for entry in self.decision_log[::-1]:
            # Entry border/indicator
            pygame.draw.line(log_surface, entry['color'], (5, y_cursor), (5, y_cursor + entry_h - 20), 3)
            
            # Time & ID
            header = f"Task-{entry['id']} | T: {entry['time']:.1f}s"
            header_surf = self.font.render(header, True, entry['color'])
            log_surface.blit(header_surf, (15, y_cursor))
            
            # Message lines
            messages = entry['msg'].split('\n')
            for i, line in enumerate(messages):
                c = WHITE if i > 0 else entry['color']
                if "Karar:" in line or "Decision:" in line: c = ACID_GREEN
                
                line_surf = self.font.render(line, True, c)
                log_surface.blit(line_surf, (20, y_cursor + 20 + (i * 14)))
            
            y_cursor += entry_h
        
        # Update max scroll
        total_content_h = y_cursor
        self.max_scroll = min(0, clip_rect.height - total_content_h)
        
        # Blit the log surface to screen with scroll offset (with CLIPPING)
        self.screen.set_clip(clip_rect)
        self.screen.blit(log_surface, (clip_rect.x, clip_rect.y + self.scroll_y))
        self.screen.set_clip(None)
        
        # Draw scroll hint if content exceeds panel
        if total_content_h > clip_rect.height:
            hint = self.font.render("↕ Scroll for history", True, GRAY)
            self.screen.blit(hint, (SIDE_PANEL_X + SIDE_PANEL_WIDTH - 120, panel_y + 15))

    def draw(self):
        if not self.running: return
        
        self.screen.fill((25, 25, 30)) # Modern Dark background
        
        # Set clipping for the central map area
        map_clip = pygame.Rect(MAP_X, 0, MAP_WIDTH, SCREEN_HEIGHT)
        self.screen.set_clip(map_clip)
        
        self.draw_grid()
        
        # Draw Cloud (Top Right in world coords, e.g., 900, 100)
        cx, cy = 900, 100
        csx = MAP_X + (cx * self.zoom_level) + self.map_offset[0]
        csy = (cy * self.zoom_level) + self.map_offset[1]
        
        if MAP_X <= csx <= SIDE_PANEL_X:
            cloud_icon = self.icon_font.render("☁️", True, BLUE)
            self.screen.blit(cloud_icon, (csx - 20, csy - 40))
            text = self.font.render("CLOUD", True, WHITE)
            self.screen.blit(text, (csx - 10, csy - 0))
            
            # Cloud queue indicator
            cloud_queue_ratio = min(1.0, self.cloud_server.queue_length / self.cloud_server.max_queue_size)
            cloud_queue_width = 80
            cloud_queue_height = 12
            cloud_queue_x, cloud_queue_y = csx - 40, csy + 20
            
            pygame.draw.rect(self.screen, DARK_GRAY, (cloud_queue_x, cloud_queue_y, cloud_queue_width, cloud_queue_height), border_radius=3)
            if cloud_queue_ratio > 0.7:
                cloud_queue_color = RED
            elif cloud_queue_ratio > 0.4:
                cloud_queue_color = ORANGE
            else:
                cloud_queue_color = BLUE
            pygame.draw.rect(self.screen, cloud_queue_color, (cloud_queue_x, cloud_queue_y, int(cloud_queue_width * cloud_queue_ratio), cloud_queue_height), border_radius=3)
            pygame.draw.rect(self.screen, CYAN, (cloud_queue_x, cloud_queue_y, cloud_queue_width, cloud_queue_height), 1, border_radius=3)
            
            cloud_queue_text = self.font.render(f"Q: {self.cloud_server.queue_length} | Load: {self.cloud_server.current_load}", True, WHITE)
            self.screen.blit(cloud_queue_text, (csx - 45, csy + 38))
        
        # Draw Edge Servers
        for edge in self.edge_servers:
            wx, wy = edge.location
            x = MAP_X + (wx * self.zoom_level) + self.map_offset[0]
            y = (wy * self.zoom_level) + self.map_offset[1]
            
            if MAP_X - 100 <= x <= SIDE_PANEL_X + 100:
                icon = self.icon_font.render("🏢", True, GREEN)
                rect = icon.get_rect(center=(x, y))
                self.screen.blit(icon, rect)
                
                # Load indicator
                load_text = self.font.render(f"E-{self.edge_servers.index(edge)} Load: {edge.current_load}", True, WHITE)
                self.screen.blit(load_text, (x-35, y+20))
            
            # Queue length bar
            queue_length = len(edge.resource.queue)
            queue_ratio = min(1.0, queue_length / edge.max_queue_size)
            queue_bar_width = 60
            queue_bar_height = 10
            queue_x, queue_y = x - 30, y + 35
            
            # Background
            pygame.draw.rect(self.screen, DARK_GRAY, (queue_x, queue_y, queue_bar_width, queue_bar_height), border_radius=3)
            if queue_ratio > 0.7:
                queue_color = RED
            elif queue_ratio > 0.4:
                queue_color = ORANGE
            else:
                queue_color = GREEN
            pygame.draw.rect(self.screen, queue_color, (queue_x, queue_y, int(queue_bar_width * queue_ratio), queue_bar_height), border_radius=3)
            pygame.draw.rect(self.screen, ACID_GREEN, (queue_x, queue_y, queue_bar_width, queue_bar_height), 1, border_radius=3)
            
            # Queue text
            queue_text = self.font.render(f"Q: {queue_length}", True, WHITE)
            self.screen.blit(queue_text, (x - 15, y + 48))
            
            # CPU frequency bar
            freq_ratio = edge.current_freq / edge.max_freq
            bar_width = 40
            bar_height = 5
            pygame.draw.rect(self.screen, GRAY, (x-20, y+58, bar_width, bar_height))
            pygame.draw.rect(self.screen, GREEN, (x-20, y+58, int(bar_width * freq_ratio), bar_height))
            
        # Draw Devices
        for device in self.devices:
            wx, wy = device.location
            x = MAP_X + (wx * self.zoom_level) + self.map_offset[0]
            y = (wy * self.zoom_level) + self.map_offset[1]
            
            if MAP_X - 50 <= x <= SIDE_PANEL_X + 50:
                # Battery status color (normalize to percentage)
                battery_pct = (device.battery / 10000.0) * 100 
                battery_pct = max(0, min(100, battery_pct))
                
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
                bg_radius = int(25 * self.zoom_level)
                if bg_radius < 5: bg_radius = 5
                pygame.draw.circle(self.screen, (240, 240, 240), (int(x), int(y)), bg_radius)  # Background
                pygame.draw.circle(self.screen, status_color, (int(x), int(y)), bg_radius, 2)  # Status ring
                
                # Device icon (scaled)
                if self.zoom_level > 0.7:
                    device_icon = self.icon_font.render("🚗", True, BLACK)
                    rect = device_icon.get_rect(center=(int(x), int(y)))
                    self.screen.blit(device_icon, rect)
                
                battery_text = self.font.render(f"{int(battery_pct)}%", True, status_color)
                self.screen.blit(battery_text, (int(x) - 15, int(y) + bg_radius + 5))
            
            # Battery bar with gradient effect (transformed)
            bar_width, bar_height = int(40 * self.zoom_level), int(6 * self.zoom_level)
            bar_x, bar_y = int(x - bar_width/2), int(y - bg_radius - bar_height - 10)
            
            if bar_width > 5:
                # Background
                pygame.draw.rect(self.screen, GRAY, (bar_x, bar_y, bar_width, bar_height))
                
                # Filled portion
                fill_width = int(bar_width * (battery_pct / 100.0))
                pygame.draw.rect(self.screen, status_color, (bar_x, bar_y, fill_width, bar_height))
                
                # Border
                pygame.draw.rect(self.screen, BLACK, (bar_x, bar_y, bar_width, bar_height), 1)
                
            # Status icon (smaller, top left of device)
            if self.zoom_level > 0.8:
                status_surf = self.small_icon_font.render(status_icon, True, status_color)
                self.screen.blit(status_surf, (int(x - bg_radius), int(y - bg_radius)))
            
            # Draw Active Links (Lines)
            if device.current_target and MAP_X <= x <= SIDE_PANEL_X:
                if hasattr(device.current_target, 'location'):
                    twx, twy = device.current_target.location
                else:
                    twx, twy = 900, 100 # Cloud
                
                tx = MAP_X + (twx * self.zoom_level) + self.map_offset[0]
                ty = (twy * self.zoom_level) + self.map_offset[1]
                
                line_color = ORANGE if device.current_task_type == "HIGH_DATA" else BLUE
                pygame.draw.line(self.screen, line_color, (int(x), int(y)), (int(tx), int(ty)), 2)

        # Update and draw particles
        for particle in self.particles[:]:
            particle.update()
            if particle.alive:
                particle.draw(self.screen, self.font, self.zoom_level, self.map_offset)
            else:
                self.particles.remove(particle)
        
        # Clear map clipping
        self.screen.set_clip(None)
        
        # Methodology Panel (Left)
        self.draw_methodology_panel()
        
        # Side Panel (Right)
        self.screen.blit(self.side_panel_surf, (SIDE_PANEL_X, 0))
        pygame.draw.line(self.screen, CYAN, (SIDE_PANEL_X, 0), (SIDE_PANEL_X, SCREEN_HEIGHT), 2)
        
        # Draw panels
        self.draw_legend()
        self.draw_knowledge_base()
        self.draw_decision_log()
        self.draw_performance_analytics()
        self.draw_toast() # Draw toast on top
        
        pygame.display.flip()
        self.handle_events()
        
    def draw_performance_analytics(self):
        """Redesigned Comparison Drawer for maximum clarity and 'Wow' factor"""
        panel_y = 810 # Slightly raised for better layout
        panel_h = 175 
        panel_w = MAP_WIDTH - 40
        drawer_rect = pygame.Rect(MAP_X + 20, panel_y, panel_w, panel_h)
        
        # Deep Navy Glassmorphism Background with slight glow
        surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        surf.fill((10, 25, 55, 230))
        self.screen.blit(surf, (MAP_X + 20, panel_y))
        pygame.draw.rect(self.screen, CYAN, drawer_rect, 1, border_radius=12)
        
        # Title with Icon
        title_surf = self.title_font.render("🏆 AI PERFORMANCE COMPARISON: WHY PPO IS BETTER?", True, ACID_GREEN)
        self.screen.blit(title_surf, (MAP_X + 50, panel_y + 15))
        
        # Data Retrieval - Using Averages to avoid extreme cumulative bias
        count = max(1, self.stats['tasks_offloaded'])
        ppo_lat_avg = self.stats['ppo_lat'] / count
        greedy_lat_avg = self.stats['greedy_lat'] / count
        
        ppo_en_avg = self.stats['ppo_en'] / count
        greedy_en_avg = self.stats['greedy_en'] / count
        
        def get_gain(ppo, base):
            if base <= 0: return 0
            # Ensure we don't show crazy numbers if samples are unbalanced
            gain = ((base - ppo) / base) * 100
            if gain < -100: return -100 # Clamp for UI sanity
            return gain
        
        lat_gain_greedy = get_gain(ppo_lat_avg, greedy_lat_avg)
        en_save_greedy = get_gain(ppo_en_avg, greedy_en_avg)
        
        # Font styling for emphasis
        gain_font = pygame.font.SysFont("Arial", 22, bold=True)
        small_bold = pygame.font.SysFont("Arial", 14, bold=True)
        med_font = pygame.font.SysFont("Arial", 16)
        
        # --- SECTION 1: TIMING PROFIT (LATENCY) ---
        col1_x = MAP_X + 40
        pygame.draw.rect(self.screen, (30, 45, 80), (col1_x - 10, panel_y + 45, 230, 110), border_radius=8)
        # Robust Icon: Blue Circle
        pygame.draw.circle(self.screen, BLUE, (col1_x + 10, panel_y + 58), 5)
        self.screen.blit(small_bold.render("ZAMANSAL KAZANÇ", True, BLUE), (col1_x + 25, panel_y + 53))
        
        prefix = "HIZLI" if lat_gain_greedy >= 0 else "YAVAŞ"
        gain_text = f"%{abs(lat_gain_greedy):.1f} {prefix}"
        g_surf = gain_font.render(gain_text, True, GREEN if lat_gain_greedy > 0 else RED)
        self.screen.blit(g_surf, (col1_x + 10, panel_y + 80))
        self.screen.blit(self.small_font.render("(vs Greedy Baseline)", True, GRAY), (col1_x + 10, panel_y + 115))
        
        # --- SECTION 2: ENERGY PROFIT (BATTERY) ---
        col2_x = col1_x + 245
        pygame.draw.rect(self.screen, (30, 45, 80), (col2_x - 10, panel_y + 45, 230, 110), border_radius=8)
        # Robust Icon: Golden Circle
        pygame.draw.circle(self.screen, GOLD, (col2_x + 10, panel_y + 58), 5)
        self.screen.blit(small_bold.render("ENERJİ TASARRUFU", True, GOLD), (col2_x + 25, panel_y + 53))
        
        prefix_en = "VERİMLİ" if en_save_greedy >= 0 else "SAVURGAN"
        en_text = f"%{abs(en_save_greedy):.1f} {prefix_en}"
        e_surf = gain_font.render(en_text, True, GREEN if en_save_greedy > 0 else RED)
        self.screen.blit(e_surf, (col2_x + 10, panel_y + 80))
        self.screen.blit(self.small_font.render("(Batarya Koruma Oranı)", True, GRAY), (col2_x + 10, panel_y + 115))
        
        # --- SECTION 3: SCIENTIFIC INSIGHT ---
        col3_x = col2_x + 245
        # Robust Icon: Orange Circle
        pygame.draw.circle(self.screen, ORANGE, (col3_x, panel_y + 58), 5)
        self.screen.blit(small_bold.render("BİLİMSEL ANALİZ", True, ORANGE), (col3_x + 15, panel_y + 53))
        
        insight_title = "PPO Avantaj Analizi:"
        if self.stats['tasks_offloaded'] < 3:
            insight_body = "Veri toplanıyor..."
            insight_foot = ""
        elif lat_gain_greedy > 5:
            insight_body = "PPO, Edge düğümlerindeki"
            insight_foot = "yoğunluğu öngördü."
        else:
            insight_body = "Cihaz batarya sağlığı için"
            insight_foot = "offload kararı optimize edildi."
            
        self.screen.blit(med_font.render(insight_title, True, WHITE), (col3_x, panel_y + 75))
        self.screen.blit(med_font.render(insight_body, True, LIGHT_GRAY), (col3_x, panel_y + 98))
        if insight_foot: self.screen.blit(med_font.render(insight_foot, True, LIGHT_GRAY), (col3_x, panel_y + 118))

        # --- SECTION 4: QoS & FAIRNESS ---
        col4_x = col3_x + 215 # Tightest spacing (Total X around 1350)
        # Robust Icon: Purple Circle
        pygame.draw.circle(self.screen, PURPLE, (col4_x, panel_y + 58), 5)
        self.screen.blit(small_bold.render("QoS & FAIRNESS", True, PURPLE), (col4_x + 15, panel_y + 53))
        
        fairness = self.stats.get('fairness_index', 1.0)
        jitter = self.stats.get('jitter_avg', 0.0)
        qoe = self.stats.get('qoe_score', 100.0)
        
        f_text = self.small_font.render(f"Jain's Fairness: {fairness:.2f}", True, LIGHT_GRAY)
        j_text = self.small_font.render(f"Avg Jitter: {jitter:.3f}s", True, LIGHT_GRAY)
        q_text = self.small_font.render(f"QoE Score: {qoe:.1f}/100", True, ACID_GREEN if qoe > 70 else ORANGE)
        
        self.screen.blit(f_text, (col4_x, panel_y + 75))
        self.screen.blit(j_text, (col4_x, panel_y + 95))
        self.screen.blit(q_text, (col4_x, panel_y + 115))

    def show_toast(self, message, duration=90):
        """Trigger a toast notification"""
        self.toast_msg = message
        self.toast_timer = duration
        self.toast_alpha = 255

    def draw_toast(self):
        """Draw fading toast notification in center"""
        if self.toast_timer > 0:
            toast_w, toast_h = 400, 60
            toast_x = MAP_X + (MAP_WIDTH - toast_w) // 2
            toast_y = 100
            
            # Fade logic
            if self.toast_timer < 20:
                self.toast_alpha = int(255 * (self.toast_timer / 20))
            
            surf = pygame.Surface((toast_w, toast_h), pygame.SRCALPHA)
            surf.fill((0, 0, 0, self.toast_alpha // 2))
            pygame.draw.rect(surf, ACID_GREEN, (0, 0, toast_w, toast_h), 2, border_radius=10)
            
            text = self.title_font.render(self.toast_msg, True, WHITE)
            text.set_alpha(self.toast_alpha)
            text_rect = text.get_rect(center=(toast_w // 2, toast_h // 2))
            surf.blit(text, text_rect)
            
            self.screen.blit(surf, (toast_x, toast_y))
            self.toast_timer -= 1

    def update(self):
        self.draw()
        self.clock.tick(30)  # 30 FPS
