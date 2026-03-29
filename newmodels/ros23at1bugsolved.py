"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PLANET FACTORY: PHASE 23 - ELECTROMAGNETIC RESONANCE          ║
║                                                                              ║
║  Integrates the theories of Michael Faraday and Nikola Tesla.                ║
║  - Faraday Induction: Visual latent flux directly induces SNN attention.     ║
║  - Tesla Resonance: Planetary grid operates on AC frequencies; out-of-phase  ║
║    energy consumption triggers cognitive warnings and resonant recharge.     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import math
import sqlite3
import csv
import random
from collections import defaultdict, deque
from dataclasses import dataclass

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Import core architecture
from ucf import (
    PerceptionPipeline, FeatureNormalizer, HoloSynHeads, 
    StudentDistilledHeadsHF, StudentDistilledHeadsBasic, ModalityBundle, 
    SafetyGovernor, IndustrialThresholds, DEVICE, NORM_JSON
)

# ── 🧠 NEUROMORPHIC BACKEND ──────────────────────────────────────────────────
try:
    import brian2 as b2
    b2.prefs.codegen.target = 'numpy'
    HAS_NEUROMORPHIC = True
except ImportError:
    HAS_NEUROMORPHIC = False

# ── 💾 PERSISTENT DATABASE ───────────────────────────────────────────────────
class DatabaseLogger:
    def __init__(self, db_path="collected_data.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_db()
    def _init_db(self):
        self.cursor.execute('CREATE TABLE IF NOT EXISTS planetary_economy (planet_id TEXT, nrg REAL, ore REAL, goods REAL)')
        self.cursor.execute('CREATE TABLE IF NOT EXISTS swarm_actions (agent_id TEXT, action TEXT, rwd REAL, temp REAL)')
        self.conn.commit()
    def log_action(self, a_id, act, rwd, temp):
        self.cursor.execute('INSERT INTO swarm_actions VALUES (?, ?, ?, ?)', (a_id, act, rwd, temp))
        self.conn.commit()

db_logger = DatabaseLogger()

# ── 📊 COSMIC DATA LAKE ──────────────────────────────────────────────────────
class CosmicDataLake:
    def __init__(self):
        self.data = []
        path = "/home/scidev/PycharmProjects/PythonProject/autocad_data.csv"
        if os.path.exists(path):
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader: self.data.append(float(row.get("Geometric Precision", 0.5)))
    def get_sense(self, cycle):
        base = self.data[cycle % len(self.data)] if self.data else 0.5
        anomaly = -0.3 if cycle % 50 == 0 else 0.0
        return max(0.1, base + anomaly)

data_lake = CosmicDataLake()

# ═════════════════════════════════════════════════════════════════════════════
#  1. MAGNETO-OPTIC VISUAL CORTEX
# ═════════════════════════════════════════════════════════════════════════════

class MagnetoProjector(nn.Module):
    """Maps raw magneto-optical 128x128 features to the 256-D latent space."""
    def __init__(self, output_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, output_dim)
        )
    def forward(self, x):
        return self.projector(self.backbone(x))

class MagnetoOpticSensor:
    """Simulates the robotic camera capturing magnetic domain materials."""
    def scan_material(self):
        is_defective = np.random.rand() < 0.2 # 20% chance of magnetic fracture
        
        if HAS_CV2:
            noise = np.random.randn(128, 128).astype(np.float32)
            sigma = np.random.uniform(2.0, 5.0)
            domain = cv2.GaussianBlur(noise, (0, 0), sigma)
            _, maze = cv2.threshold(domain, 0, 1, cv2.THRESH_BINARY)
            if is_defective: cv2.circle(maze, (64, 64), 25, 0, -1) # Dead zone
            maze = (maze * 255).astype(np.uint8)
            maze = cv2.GaussianBlur(maze, (3, 3), 0)
            gauss = np.random.normal(0, 15, (128, 128)).astype(np.float32)
            optic_img = np.clip(maze.astype(np.float32) + gauss, 0, 255) / 255.0
            tensor = torch.from_numpy(optic_img).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        else:
            tensor = torch.randn(1, 1, 128, 128).to(DEVICE)
            if is_defective: tensor[:, :, 50:78, 50:78] = 0.0
            
        return tensor, is_defective

# ═════════════════════════════════════════════════════════════════════════════
#  2. HYBRID INTELLIGENCE & SNN
# ═════════════════════════════════════════════════════════════════════════════

class PyTorchTeacherHead(nn.Module):
    def __init__(self, out_dim=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, out_dim))
    def forward(self, x): return self.net(x)

class OnlineNeuromorphicHead:
    def __init__(self, n_actions=5):
        if not HAS_NEUROMORPHIC: return
        b2.start_scope()
        self.P = b2.PoissonGroup(256, rates=np.zeros(256)*b2.Hz)
        self.G = b2.NeuronGroup(n_actions, 'dv/dt = (I - v) / (10*ms) : 1\nI : 1', threshold='v > 1.0', reset='v=0', method='euler')
        eqs = (
            'dw/dt = (1.5*Hz * attention) * reward - 0.01*Hz * w : 1 (clock-driven)\n'
            'reward : 1 (shared)\n'
            'attention : 1 (shared)'
        )
        self.S = b2.Synapses(self.P, self.G, model=eqs, on_pre='v_post += w', method='euler')
        self.S.connect(); self.S.w = 'rand()*0.2'
        self.M = b2.SpikeMonitor(self.G); self.net = b2.Network(self.P, self.G, self.S, self.M)
        self.last = np.zeros(n_actions)
    def predict_and_learn(self, z, r, a):
        self.P.rates = np.clip(z.squeeze().cpu().numpy(), 0, 1)*100*b2.Hz
        self.S.reward, self.S.attention = r, a; self.net.run(20*b2.ms)
        c = np.array(self.M.count) - self.last; self.last = np.array(self.M.count)
        return int(np.argmax(c)) if np.sum(c) > 0 else random.randint(0, 4), 0.02

# ═════════════════════════════════════════════════════════════════════════════
#  3. MASTER SWARM AGENT (TESLA-FARADAY EDITION)
# ═════════════════════════════════════════════════════════════════════════════

class ApexSwarmNode:
    def __init__(self, name, chassis, planet_id):
        self.name, self.chassis, self.planet_id = name, chassis, planet_id
        self.energy = 10000.0; self.temp = 35.0; self.avg_rwd = 0.0
        
        self.head = OnlineNeuromorphicHead(n_actions=5)
        self.camera = MagnetoOpticSensor()
        self.visual_cortex = MagnetoProjector(output_dim=256).to(DEVICE)
        
        search_dirs = [os.getcwd(), os.path.dirname(os.path.abspath(__file__)), "/home/scidev/PycharmProjects/PythonProject/"]
        
        # Load Perfected SNN Weights
        w_path = os.path.join("/home/scidev/PycharmProjects/PythonProject/", "optimized_living_planet_weights.pt")
        if os.path.exists(w_path):
            try:
                data = torch.load(w_path, map_location='cpu', weights_only=False)
                self.head.S.w = data['synaptic_weights']
            except Exception: pass

        # Load Visual Weights
        for d in search_dirs:
            p = os.path.join(d, "magneto_projector_weights.pt")
            if os.path.exists(p):
                self.visual_cortex.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=False))
                break
                
        # Load Teacher Instincts
        self.has_teacher = False
        t_name = "robot_arm_snn_head.pt" if chassis in ["Borer", "Assembler"] else "cnc_production_snn_head.pt"
        for d in search_dirs:
            p = os.path.join(d, t_name)
            if os.path.exists(p):
                try:
                    checkpoint = torch.load(p, map_location=DEVICE, weights_only=False)
                    out_dim = checkpoint['net.2.bias'].shape[0] if 'net.2.bias' in checkpoint else 5
                    self.teacher = PyTorchTeacherHead(out_dim=out_dim).to(DEVICE)
                    self.teacher.load_state_dict(checkpoint, strict=False)
                    self.teacher.eval()
                    self.has_teacher = True
                    self.teacher_out_dim = out_dim
                    break
                except Exception: pass

        self.perception = PerceptionPipeline(FeatureNormalizer(NORM_JSON), HoloSynHeads(), StudentDistilledHeadsHF(), StudentDistilledHeadsBasic())
        self.governor = SafetyGovernor(IndustrialThresholds(max_temp=120.0))

    def step(self, cycle):
        # ── 1. SENSORY INPUT ──
        prec = data_lake.get_sense(cycle)
        feat = torch.tensor([[prec, 0.8, self.energy/10000.0, 0.1, 0.1]], dtype=torch.float32).to(DEVICE)
        bundle = ModalityBundle(E_t_aug=torch.randn(1, 528), E_i=torch.zeros(1, 2048), E_a=torch.randn(1, 2048), E_v=torch.randn(1, 2048), H=torch.zeros(1, 32, 3), feat_hf=torch.randn(1, 789), feat_basic=feat, mask=torch.ones(1, 4))
        
        econ_latent, _, _ = self.perception.perceive(bundle)
        img_tensor, is_defective = self.camera.scan_material()
        
        with torch.no_grad():
            vis_latent = self.visual_cortex(img_tensor)
            
        fused_latent = (econ_latent + vis_latent) / 2.0

        # ── 2. TESLA-FARADAY ATTENTION MECHANICS ──
        # NIKOLA TESLA: Alternating Current (AC) Resonance 
        # The planetary grid oscillates. Being out of phase causes high attention (stress).
        tesla_resonance = math.sin(cycle * math.pi / 12.0) 
        grid_phase_error = abs((self.energy / 10000.0) - (0.5 + 0.5 * tesla_resonance))
        
        # MICHAEL FARADAY: Magnetic Induction
        # High variance in the magneto-optic latent space explicitly induces cognitive focus.
        faraday_flux = torch.std(vis_latent).item() * 5.0 

        # Standard Reward Prediction Error
        rwd_signal = (prec - 0.5) * 2.0
        rpe = abs(rwd_signal - self.avg_rwd)
        self.avg_rwd = 0.9 * self.avg_rwd + 0.1 * rwd_signal
        
        # FUSED ATTENTION SIGNAL (Analog, continuously shifting)
        attn = 1.0 + (3.0 * rpe) + (4.0 * grid_phase_error) + faraday_flux

        # ── 3. COGNITION ──
        teacher_action = 4
        if self.has_teacher:
            with torch.no_grad():
                t_logits = self.teacher(fused_latent)
                padded_logits = torch.full((t_logits.shape[0], 5), -float('inf')).to(DEVICE)
                padded_logits[:, :self.teacher_out_dim] = t_logits
                teacher_action = torch.argmax(padded_logits, dim=-1).item()

        action, lat = self.head.predict_and_learn(fused_latent, rwd_signal, attn)
        if self.has_teacher and random.random() < 0.1: action = teacher_action
        
        # ── 4. ACT, REWARD & AC RESONANCE ──
        act_name = ["Mine", "Mfg", "Asm", "Logi", "Maint"][action]
        final_reward = rwd_signal
        
        if action == 4: 
            # TESLA RESONANT CHARGING: Recharging during peak AC flux yields bonus energy
            ac_boost = max(0.0, tesla_resonance * 100.0)
            self.energy += (300 + ac_boost) 
            self.temp -= 15
            
            if self.energy > 8000: final_reward -= 2.0 
            else: final_reward += 1.0 + (tesla_resonance * 0.5) # Resonant reward
        else: 
            self.energy -= 100; self.temp += 10
            if action in [1, 2]: 
                if is_defective:
                    final_reward -= 5.0
                    attn += 5.0 # Sudden fracture shock
                    act_name += " ❌(DEFECT)"
                else: final_reward += 2.0 
            elif action in [0, 3]: final_reward += 1.0 
            
        self.energy = max(0, min(10000, self.energy))
        self.head.predict_and_learn(fused_latent, final_reward, attn)
        
        status_color = "\033[91m" if "DEFECT" in act_name else "\033[96m" if action == 4 and tesla_resonance > 0.5 else "\033[92m"
        print(f"[{self.name}] Act: {status_color}{act_name:<16}\033[0m | NRG: {self.energy:4.0f} | Attn: {attn:5.2f} (Flux: {faraday_flux:.2f}) | Rwd: {final_reward:+.1f}")
        db_logger.log_action(self.name, act_name, final_reward, self.temp)

# ── MAIN DEPLOYMENT ──
if __name__ == "__main__":
    print("\n" + "═"*90 + "\n  ⚡ PHASE 23: TESLA-FARADAY ELECTROMAGNETIC RESONANCE PROTOCOL\n" + "═"*90)
    swarm = [ApexSwarmNode("Alpha-1", "Assembler", "Vulcan"), ApexSwarmNode("Beta-1", "Foundry", "Vulcan")]
    
    for cycle in range(1, 101):
        if cycle % 20 == 0: print("-" * 75)
        for drone in swarm: drone.step(cycle)
        time.sleep(0.02)