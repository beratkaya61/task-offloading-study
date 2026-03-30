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
    def __init__(self, start_pos, end_pos, color, task_id, is_partial=False):
        self.pos = list(start_pos)
        self.target = end_pos
        self.color = color
        self.task_id = task_id
        self.speed = 7
        self.alive = True
        self.is_partial = is_partial
        
    def update(self):
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
        
        if MAP_X <= sx <= SIDE_PANEL_X:
            # For partial offloading, draw a 'ghost' trail or split look
            if self.is_partial:
                pygame.draw.circle(screen, self.color, (int(sx)-3, int(sy)-3), int(4 * zoom_level), 1)
            
            pygame.draw.circle(screen, self.color, (int(sx), int(sy)), int(5 * zoom_level))
            
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
        self.feed_bold_font = pygame.font.SysFont("Arial", 14, bold=True) # For feedback labels
        self.json_font = pygame.font.SysFont("Consolas", 13) # Monospaced for JSON
        
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
        
        # ✅ Mouse Drag (Pan) State for Map
        self.is_dragging = False
        self.drag_start = None
        self.drag_start_offset = None
        
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
            
            # ✅ Mouse Drag (Pan) for Map Navigation
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                # Enable drag only on map area (not on side panels)
                if METHOD_PANEL_WIDTH < mx < SIDE_PANEL_X:
                    self.is_dragging = True
                    self.drag_start = (mx, my)
                    self.drag_start_offset = list(self.map_offset)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                self.is_dragging = False
                self.drag_start = None
                self.drag_start_offset = None
            
            elif event.type == pygame.MOUSEMOTION and self.is_dragging and self.drag_start and self.drag_start_offset:
                mx, my = pygame.mouse.get_pos()
                dx = mx - self.drag_start[0]
                dy = my - self.drag_start[1]
                
                # Update offset based on drag
                self.map_offset[0] = self.drag_start_offset[0] + dx
                self.map_offset[1] = self.drag_start_offset[1] + dy

    def add_task_particle(self, start_pos, end_pos, color, task_id, is_partial=False):
        """Add animated particle for task flow with ID"""
        self.particles.append(TaskParticle(start_pos, end_pos, color, task_id, is_partial))

    def add_decision_log(self, task_id, message, color=BLACK, is_json=False, metadata=None):
        """Add a detailed decision reasoning with optional JSON metadata"""
        if not hasattr(self, 'ppo_action_history'): self.ppo_action_history = []
        
        if is_json and message:
            try:
                import json
                data = json.loads(message)
                action_str = data.get("ppo_logic", {}).get("action", "ACTION_0")
                action_id = int(action_str.split("_")[-1])
                self.ppo_action_history.append(action_id)
                if len(self.ppo_action_history) > 100: self.ppo_action_history.pop(0)
            except: pass

        self.decision_log.append({
            "id": task_id, 
            "msg": message, 
            "color": color, 
            "time": self.env.now,
            "is_json": is_json,
            "metadata": metadata
        })
        if len(self.decision_log) > 100:
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
        
        # ✅ SIMPLIFIED LAYOUT: Remove Infrastructure Status, consolidate to Offload Distribution
        
        # --- Performance Metrics Card (Top) ---
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

        # Detailed Breakdown - Offload Distribution (Main section)
        y_offset += 15
        breakdown_title = self.font.render("OFFLOAD DISTRIBUTION", True, CYAN)
        self.screen.blit(breakdown_title, (SIDE_PANEL_X + 35, y_offset))
        y_offset += 20
        
        # Get action counts from stats
        action_counts = self.stats.get('action_counts', {i: 0 for i in range(6)})
        action_labels = {
            0: "Full Local",
            1: "Partial Local 75% Edge 25%",
            2: "Partial Local 50% Edge 50%",
            3: "Partial Local 25% Edge 75%",
            4: "Full Edge",
            5: "Full Cloud"
        }
        
        # Show all action distributions
        for action_id in range(6):
            count = action_counts.get(action_id, 0)
            label = action_labels.get(action_id, f"Action {action_id}")
            
            # Color coding based on action type
            if action_id == 0:
                color = GOLD  # Full Local - battery saving
            elif 1 <= action_id <= 3:
                color = (150, 200, 100)  # Partial - balanced
            elif action_id == 4:
                color = GREEN  # Edge - moderate offload
            else:  # 5
                color = BLUE  # Cloud - full offload
            
            task_surf = self.small_font.render(f"{action_id}: {label}: {count}", True, color)
            self.screen.blit(task_surf, (SIDE_PANEL_X + 45, y_offset))
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
                "Yöntem: Shannon-Hartley modeli.",
                "Uygulama: Cihaz ve Edge server arasındaki",
                "kablosuz kanal kapasitesini belirler.",
                "Bilimsel Veri: SNR ve mesafe bazlı anlık",
                "iletişim dalgalanmalarını hesaplar."
            ]),
            ("ENERJİ (DVFS Modeli)", CYAN, [
                "Yöntem: Dinamik Voltaj ve Frekans Ölçekleme.",
                "Amaç: Edge Server işlem gücü verimliliğini",
                "optimize etmek için bu tarafta modellenir.",
                "Not: Bulut tarafı sınırsız kaynak kabul",
                "edildiği için enerji modellemesi gerekmez."
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

        # --- LLM vs RULE-BASED USAGE STATS ---
        y += 20
        pygame.draw.circle(self.screen, ACID_GREEN, (25, y+10), 5)
        llm_title = sub_header_font.render("SEMANTIC ANALYZER", True, ACID_GREEN)
        self.screen.blit(llm_title, (40, y))
        y += 25
        
        # Get LLM stats from simulator
        llm_success = self.stats.get('llm_success_count', 0)
        rule_based_fallback = self.stats.get('rule_based_fallback_count', 0)
        total_analyses = llm_success + rule_based_fallback
        
        if total_analyses > 0:
            llm_rate = (llm_success / total_analyses) * 100
        else:
            llm_rate = 0
        
        # Display stats
        llm_stat = self.small_font.render(f"✓ LLM Success: {llm_success} ({llm_rate:.1f}%)", True, ACID_GREEN)
        fallback_stat = self.small_font.render(f"↺ Rule-Based: {rule_based_fallback}", True, ORANGE)
        total_stat = self.small_font.render(f"Total: {total_analyses}", True, WHITE)
        
        self.screen.blit(llm_stat, (35, y))
        y += 18
        self.screen.blit(fallback_stat, (35, y))
        y += 18
        self.screen.blit(total_stat, (35, y))
        y += 18
            
    def draw_knowledge_base(self):
        """✅ Redesigned panel for Node-specific Health Status - positioned after Offload Distribution"""
        # Position adjusted after removing Infrastructure Status
        # Now shows right after action counts (around y=300 area)
        panel_y = 350  # Adjusted: was 380 when Infrastructure Status was above
        panel_h = 145  # Slimmed down
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
            e_health = self.font.render(f"Edge-{edge.id}: {e_status} (Queue Length: {q})", True, WHITE)
            self.screen.blit(e_health, (SIDE_PANEL_X + 65, y))
            y += 20

    def draw_decision_log(self):
        """Draw Professional AI Decision Feed as requested in Screenshot"""
        # ✅ Adjusted: after NODE HEALTH STATUS moved down, feed starts at 515 (was 535)
        panel_y = 515
        panel_h = 465  # Increased height to use available space better
        
        self.screen.blit(self.log_panel_surf, (SIDE_PANEL_X + 10, panel_y))
        pygame.draw.rect(self.screen, CYAN, (SIDE_PANEL_X + 10, panel_y, SIDE_PANEL_WIDTH - 20, panel_h), 1, border_radius=5)
        
        title_surf = self.title_font.render("SEMANTIC DECISION FEED", True, ACID_GREEN)
        self.screen.blit(title_surf, (SIDE_PANEL_X + 25, panel_y + 15))
        
        # ✅ Increased clip area: from (panel_h - 60) to (panel_h - 50) for better visibility
        clip_rect = pygame.Rect(SIDE_PANEL_X + 15, panel_y + 50, SIDE_PANEL_WIDTH - 30, panel_h - 50)
        log_surface = pygame.Surface((clip_rect.width, 8000), pygame.SRCALPHA)  # Increased surface height
        
        y_cursor = 0
        entry_sep = 180 # Reduced for compact metadata display (more entries visible)
        max_w = clip_rect.width - 40
        
        for entry in self.decision_log[::-1]:
            # Start of entry block
            entry_start_y = y_cursor
            
            # Header
            header = f"Task-{entry['id']} | T: {entry['time']:.1f}s"
            log_surface.blit(self.small_font.render(header, True, entry['color']), (15, y_cursor))
            y_cursor += 20
            
            # Main Analysis with Wrapping
            raw_lines = entry['msg'].split('\n')
            draw_y = y_cursor
            
            for line in raw_lines:
                c = WHITE
                font = self.font # Use larger font for general readability
                
                if "LLM Analizi:" in line: 
                    c = CYAN
                    font = self.feed_bold_font
                elif "PPO Karar:" in line: 
                    c = ACID_GREEN
                    font = self.feed_bold_font
                elif "Metod:" in line: 
                    c = GOLD
                    font = self.feed_bold_font
                elif "⚠️ Nöral Öncelik:" in line: 
                    c = (255, 150, 50)
                    font = self.feed_bold_font
                
                # Manual Wrap
                words = line.split(' ')
                current_line = ""
                for word in words:
                    test_line = current_line + word + " "
                    if font.size(test_line)[0] < max_w:
                        current_line = test_line
                    else:
                        line_surf = font.render(current_line, True, c)
                        log_surface.blit(line_surf, (20, draw_y))
                        draw_y += 15 # Adjusted for larger font
                        current_line = word + " "
                
                line_surf = font.render(current_line, True, c)
                log_surface.blit(line_surf, (20, draw_y))
                draw_y += 16
            
            # Metadata JSON - ENHANCED for full visibility
            metadata_y = draw_y + 10  # Space after text
            metadata_end_y = draw_y  # Will be updated if metadata exists
            if entry.get('metadata'):
                try:
                    meta = entry['metadata']
                    
                    # Create formatted JSON lines for display - COMPLETE FIELDS
                    # Use larger font for JSON
                    json_font = self.font  # Larger than small_font
                    max_width = clip_rect.width - 80  # Leave margin for wrapping
                    
                    json_lines = []
                    json_lines.append("{")
                    json_lines.append(f'  "task_id": {meta.get("task_id", "?")},')
                    json_lines.append(f'  "priority": {meta.get("priority", "?")} [{meta.get("priority_label", "?")}],')
                    json_lines.append(f'  "urgency": {meta.get("urgency", "?")}, "complexity": {meta.get("complexity", "?")},')
                    json_lines.append(f'  "bandwidth_need": {meta.get("bandwidth_need", "?")},')
                    json_lines.append(f'  "action": "{meta.get("action", "?")}" → "{meta.get("node", "?")}",')
                    # sync field - show alignment between LLM recommendation and PPO decision
                    sync_val = meta.get("sync", "?")
                    sync_desc = "LLM↔PPO Aligned" if sync_val == "ALIGNED" else "LLM↔PPO Conflict"
                    json_lines.append(f'  "sync": "{sync_desc}", "llm_recommendation": "{meta.get("llm_recommendation", "?")}",')
                    
                    # Handle reason field with smart wrapping
                    reason_full = meta.get("reason", "N/A")
                    if len(reason_full) > 70:
                        # Split into chunks that fit the box width
                        words = reason_full.split()
                        current_line = '  "reason": "'
                        reason_lines = []
                        for word in words:
                            test_line = current_line + word + " "
                            test_surf = json_font.render(test_line, True, (200, 200, 200))
                            if test_surf.get_width() < max_width:
                                current_line = test_line
                            else:
                                reason_lines.append(current_line.rstrip() + '\n')
                                current_line = '             ' + word + " "
                        reason_lines.append(current_line.rstrip() + '",')
                        json_lines.extend(reason_lines)
                    else:
                        json_lines.append(f'  "reason": "{reason_full}",')
                    
                    # Stats sub-object - ALL FIELDS
                    stats = meta.get('stats', {})
                    json_lines.append(f'  "stats": {{')
                    json_lines.append(f'    "snr_db": {stats.get("snr_db", "?")}, "lat_ms": {stats.get("lat_ms", "?")},')
                    json_lines.append(f'    "size_mb": {stats.get("size_mb", "?")}, "cpu_ghz": {stats.get("cpu_ghz", "?")},')
                    json_lines.append(f'    "battery_pct": {stats.get("battery_pct", "?")}%, "edge_queue": {stats.get("edge_queue", "?")},')
                    json_lines.append(f'    "deadline_s": {stats.get("deadline_s", "?")}, "task_type": "{stats.get("task_type", "?")}"')
                    json_lines.append(f'  }}')
                    json_lines.append("}")
                    
                    # Calculate box height - ADJUSTED FOR ACTUAL LINE COUNT
                    box_height = len(json_lines) * 16 + 25
                    
                    # Draw background box with enhanced border
                    pygame.draw.rect(log_surface, (15, 25, 35), (20, metadata_y - 5, clip_rect.width - 40, box_height), border_radius=4)
                    pygame.draw.rect(log_surface, (100, 140, 160), (20, metadata_y - 5, clip_rect.width - 40, box_height), 2, border_radius=4)
                    
                    # Render JSON lines with syntax highlighting - CLEANER
                    json_y = metadata_y + 8
                    for line in json_lines:
                        # Skip empty lines from wrapping
                        if not line.strip():
                            json_y += 16
                            continue
                        
                        # Color based on content
                        if line.startswith("{") or line.startswith("}"):
                            color = (180, 180, 180)  # Gray for braces
                            line_surf = json_font.render(line, True, color)
                            log_surface.blit(line_surf, (35, json_y))
                        else:
                            # For key-value pairs, split smartly at first ": "
                            if '": ' in line:
                                key_idx = line.find('": ')
                                if key_idx != -1:
                                    key_part = line[:key_idx + 2]  # Include the ":"
                                    val_part = line[key_idx + 2:]
                                    
                                    # Render key (cyan)
                                    key_surf = json_font.render(key_part, True, (120, 200, 220))
                                    log_surface.blit(key_surf, (35, json_y))
                                    
                                    # Render value (gold)
                                    val_surf = json_font.render(val_part, True, (200, 180, 100))
                                    log_surface.blit(val_surf, (35 + key_surf.get_width(), json_y))
                                else:
                                    # Fallback: render as white
                                    line_surf = json_font.render(line, True, (200, 200, 200))
                                    log_surface.blit(line_surf, (35, json_y))
                            else:
                                # Other lines (no key-value)
                                color = (200, 200, 200)  # White for other
                                line_surf = json_font.render(line, True, color)
                                log_surface.blit(line_surf, (35, json_y))
                        
                        json_y += 16  # Fixed line height for larger font
                    
                    # Update draw_y for next entry
                    draw_y = metadata_y + box_height
                    metadata_end_y = draw_y
                    
                except Exception as e:
                    import traceback
                    print(f"[ERROR] Metadata render failed: {e}")
                    traceback.print_exc()
            else:
                # No metadata, use current draw_y
                metadata_end_y = draw_y
            
            # Add vertical indicator line - now using actual height
            entry_height = metadata_end_y - entry_start_y + 15
            pygame.draw.line(log_surface, entry['color'], (5, entry_start_y), (5, entry_start_y + entry_height), 3)
            
            # Move to next entry with spacing
            y_cursor = entry_start_y + entry_height + 20
        
        total_content_h = y_cursor
        self.max_scroll = min(0, clip_rect.height - total_content_h)
        
        self.screen.set_clip(clip_rect)
        self.screen.blit(log_surface, (clip_rect.x, clip_rect.y + self.scroll_y))
        self.screen.set_clip(None)
        
        if total_content_h > clip_rect.height:
            hint = self.small_font.render("↕ Scroll for history", True, GRAY)
            self.screen.blit(hint, (SIDE_PANEL_X + SIDE_PANEL_WIDTH - 150, panel_y + 15))

    def draw(self):
        if not self.running: return
        
        self.screen.fill((25, 25, 30)) # Modern Dark background
        
        # Set clipping for the central map area
        map_clip = pygame.Rect(MAP_X, 0, MAP_WIDTH, SCREEN_HEIGHT)
        self.screen.set_clip(map_clip)
        
        self.draw_grid()
        
        # Draw Cloud
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
            
            cloud_queue_text = self.font.render(f"Queue Length: {self.cloud_server.queue_length} | Load: {self.cloud_server.current_load}", True, WHITE)
            self.screen.blit(cloud_queue_text, (csx - 65, csy + 38))
        
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
                load_text = self.font.render(f"Edge-{edge.id} Load: {edge.current_load}", True, WHITE)
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
            queue_text = self.font.render(f"Queue Length: {queue_length}", True, WHITE)
            self.screen.blit(queue_text, (x - 25, y + 48))
            
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
                battery_pct = max(0, min(100, (device.battery / 10000.0) * 100))
                
                status_color = GREEN
                status_icon = "✓"
                if battery_pct < 20:
                    status_color = RED
                    status_icon = "⚠️"
                elif battery_pct < 50:
                    status_color = ORANGE
                    status_icon = "⚡"
                
                # Draw larger device icon with background circle
                bg_radius = int(25 * self.zoom_level)
                if bg_radius < 5: bg_radius = 5
                pygame.draw.circle(self.screen, (240, 240, 240), (int(x), int(y)), bg_radius)
                pygame.draw.circle(self.screen, status_color, (int(x), int(y)), bg_radius, 2)
                
                if self.zoom_level > 0.7:
                    device_icon = self.icon_font.render("🚗", True, BLACK)
                    rect = device_icon.get_rect(center=(int(x), int(y)))
                    self.screen.blit(device_icon, rect)
                
                battery_text = self.font.render(f"{int(battery_pct)}%", True, status_color)
                self.screen.blit(battery_text, (int(x) - 15, int(y) + bg_radius + 5))
                # Battery bar
                bar_width, bar_height = int(40 * self.zoom_level), int(6 * self.zoom_level)
                if bar_width > 5:
                    bar_x, bar_y = int(x - bar_width/2), int(y - bg_radius - bar_height - 10)
                    pygame.draw.rect(self.screen, GRAY, (bar_x, bar_y, bar_width, bar_height))
                    fill_width = int(bar_width * (battery_pct / 100.0))
                    pygame.draw.rect(self.screen, status_color, (bar_x, bar_y, fill_width, bar_height))
                    pygame.draw.rect(self.screen, BLACK, (bar_x, bar_y, bar_width, bar_height), 1)
                    
                if self.zoom_level > 0.8:
                    status_surf = self.small_icon_font.render(status_icon, True, status_color)
                    self.screen.blit(status_surf, (int(x - bg_radius), int(y - bg_radius)))
                
                if device.current_target:
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
        
        # Data Retrieval
        count = max(1, self.stats['tasks_offloaded'])
        ppo_lat_avg = self.stats.get('partial_lat', 0) / count
        binary_lat_avg = self.stats.get('binary_lat', 0) / count
        
        ppo_en_avg = self.stats.get('ppo_en', 0) / count
        greedy_en_avg = self.stats.get('greedy_en', 1.0) / count
        
        def get_gain(ppo, base):
            if base <= 0: return 0
            gain = ((base - ppo) / base) * 100
            if gain < -100: return -100
            return gain
        
        lat_gain = get_gain(ppo_lat_avg, binary_lat_avg)
        en_save = get_gain(ppo_en_avg, greedy_en_avg)
        
        # Font styling for emphasis
        gain_font = pygame.font.SysFont("Arial", 22, bold=True)
        small_bold = pygame.font.SysFont("Arial", 14, bold=True)
        med_font = pygame.font.SysFont("Arial", 16)
        
        # --- SECTION 1: ZAMANSAL KAZANÇ (LATENCY) ---
        col1_x = MAP_X + 40
        pygame.draw.rect(self.screen, (30, 45, 80), (col1_x - 10, panel_y + 45, 230, 110), border_radius=8)
        pygame.draw.circle(self.screen, BLUE, (col1_x + 10, panel_y + 58), 5)
        self.screen.blit(small_bold.render("ZAMANSAL KAZANÇ", True, BLUE), (col1_x + 25, panel_y + 53))
        
        prefix = "HIZLI" if lat_gain >= 0 else "YAVAŞ"
        gain_text = f"%{abs(lat_gain):.1f} {prefix}"
        g_surf = gain_font.render(gain_text, True, GREEN if lat_gain > 0 else RED)
        self.screen.blit(g_surf, (col1_x + 10, panel_y + 80))
        self.screen.blit(self.small_font.render("(vs Greedy Baseline)", True, GRAY), (col1_x + 10, panel_y + 115))
        
        # --- SECTION 2: ENERJİ TASARRUFU (BATTERY) ---
        col2_x = col1_x + 245
        pygame.draw.rect(self.screen, (30, 45, 80), (col2_x - 10, panel_y + 45, 230, 110), border_radius=8)
        pygame.draw.circle(self.screen, GOLD, (col2_x + 10, panel_y + 58), 5)
        self.screen.blit(small_bold.render("ENERJİ TASARRUFU", True, GOLD), (col2_x + 25, panel_y + 53))
        
        prefix_en = "VERİMLİ" if en_save >= 0 else "SAVURGAN"
        en_text = f"%{abs(en_save):.1f} {prefix_en}"
        e_surf = gain_font.render(en_text, True, GREEN if en_save > 0 else RED)
        self.screen.blit(e_surf, (col2_x + 10, panel_y + 80))
        self.screen.blit(self.small_font.render("(Batarya Koruma Oranı)", True, GRAY), (col2_x + 10, panel_y + 115))
        
        # --- SECTION 3: BİLİMSEL ANALİZ ---
        col3_x = col2_x + 245
        pygame.draw.circle(self.screen, ORANGE, (col3_x, panel_y + 58), 5)
        self.screen.blit(small_bold.render("BİLİMSEL ANALİZ", True, ORANGE), (col3_x + 15, panel_y + 53))
        
        insight_title = "PPO Avantaj Analizi:"
        insight_body = "Cihaz batarya sağlığı için"
        insight_foot = "offload kararı optimize edildi."
            
        self.screen.blit(med_font.render(insight_title, True, WHITE), (col3_x, panel_y + 75))
        self.screen.blit(med_font.render(insight_body, True, LIGHT_GRAY), (col3_x, panel_y + 98))
        if insight_foot: self.screen.blit(med_font.render(insight_foot, True, LIGHT_GRAY), (col3_x, panel_y + 118))

        # --- SECTION 4: QoS & FAIRNESS ---
        col4_x = col3_x + 215
        pygame.draw.circle(self.screen, PURPLE, (col4_x, panel_y + 58), 5)
        self.screen.blit(small_bold.render("QoS & FAIRNESS", True, PURPLE), (col4_x + 15, panel_y + 53))
        
        fairness = self.stats.get('fairness_index', 1.0)
        qoe = self.stats.get('qoe_score', 100.0)
        
        self.screen.blit(self.small_font.render(f"Jain's Fairness: {fairness:.2f}", True, LIGHT_GRAY), (col4_x, panel_y + 75))
        self.screen.blit(self.small_font.render(f"Avg Jitter: {self.stats.get('jitter_avg', 0):.3f}s", True, LIGHT_GRAY), (col4_x, panel_y + 95))
        self.screen.blit(self.small_font.render(f"QoE Score: {qoe:.1f}/100", True, ACID_GREEN if qoe > 70 else ORANGE), (col4_x, panel_y + 115))

    def draw_ppo_strategy_monitor(self, x, y):
        """Draw PPO Strategy Trend Indicator"""
        width, height = 300, 100
        s = pygame.Surface((width, height), pygame.SRCALPHA)
        s.fill((15, 25, 45, 180))
        self.screen.blit(s, (x, y))
        pygame.draw.rect(self.screen, CYAN, (x, y, width, height), 1, border_radius=10)
        
        self.screen.blit(self.small_font.render("PPO STRATEGY MONITOR", True, ACID_GREEN), (x + 10, y + 10))
        
        if not hasattr(self, 'ppo_action_history') or not self.ppo_action_history:
            self.screen.blit(self.small_font.render("Eğitim verisi bekleniyor...", True, GRAY), (x + 10, y + 40))
            return

        last_10 = self.ppo_action_history[-10:]
        local_c = last_10.count(0)
        partial_c = sum(1 for a in last_10 if 1 <= a <= 3)
        edge_c = sum(1 for a in last_10 if a >= 4)
        
        msg, clr = "KARMA (PARTIAL)", ACID_GREEN
        if local_c > 5: msg, clr = "BATARYA KORUMA", GOLD
        elif edge_c > 5: msg, clr = "PERFORMANS", BLUE
             
        self.screen.blit(self.font.render(msg, True, clr), (x + 10, y + 35))
        for i, action in enumerate(last_10):
            pygame.draw.circle(self.screen, clr, (x + 20 + i*25, y + 80 - action*6), 3)

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
