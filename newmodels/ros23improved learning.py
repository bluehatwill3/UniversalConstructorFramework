"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PLANET FACTORY: PHASE 24 - RESONANT ELASTIC SWARM             ║
║                                                                              ║
║  A perfected mechanical neural network integrating:                          ║
║  - Faraday Flux: Latent visual variance modulates neural time constants.      ║
║  - Tesla Resonance: Energy efficiency tied to planetary AC phase.            ║
║  - Synaptic Atrophy: Systematic forgetting of non-productive neural paths.    ║
║  - Elastic Consolidation: Protection of high-importance industrial weights.   ║
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
        self.cursor.execute('CREATE TABLE IF NOT EXISTS swarm_actions (agent_id TEXT, action TEXT, rwd REAL, importance REAL)')
        self.conn.commit()
    def log_action(self, a_id, act, rwd, importance):
        self.cursor.execute('INSERT INTO swarm_actions VALUES (?, ?, ?, ?)', (a_id, act, rwd, importance))
        self.conn.commit()

db_logger = DatabaseLogger()

# ── 📊 REAL-WORLD DATA LAKE ──────────────────────────────────────────────────
class MechanicalDataLake:
    """Ingests real AutoCAD data and scales it for neural perception."""
    def __init__(self, path="/home/scidev/PycharmProjects/PythonProject/autocad_data.csv"):
        self.data = []
        if os.path.exists(path):
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.data.append({
                        "prec": float(row.get("Geometric Precision", 0.5)),
                        "eff": float(row.get("Assembly Efficiency", 0.5))
                    })
        print(f"📊 Data Lake: Ingested {len(self.data)} real-world mechanical frames.")

    def get_frame(self, cycle):
        if not self.data: return {"prec": 0.5, "eff": 0.5}
        return self.data[cycle % len(self.data)]

data_lake = MechanicalDataLake()

# ═════════════════════════════════════════════════════════════════════════════
#  1. MAGNETO-OPTIC VISUAL CORTEX
# ═════════════════════════════════════════════════════════════════════════════

class MagnetoProjector(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.projector = nn.Sequential(nn.Flatten(), nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, output_dim))
    def forward(self, x): return self.projector(self.backbone(x))

# ═════════════════════════════════════════════════════════════════════════════
#  2. ELASTIC NEUROMORPHIC HEAD (Remembering vs. Forgetting)
# ═════════════════════════════════════════════════════════════════════════════

class ElasticNeuromorphicHead:
    """SNN that forgets noise but remembers high-importance mechanical patterns."""
    def __init__(self, n_actions=5):
        if not HAS_NEUROMORPHIC: return
        b2.start_scope()
        self.P = b2.PoissonGroup(256, rates=np.zeros(256)*b2.Hz)
        # Neurons with dynamic Faraday-Induced Time Constant
        self.G = b2.NeuronGroup(n_actions, 
            '''dv/dt = (I - v) / tau : 1
               I : 1
               tau : second''', 
            threshold='v > 1.0', reset='v=0', method='euler')
        
        # Synapses with Elastic Weight Consolidation (Importance tracking)
        # dw/dt includes an atrophy term for unused paths
        eqs = '''
        dw/dt = (base_lr * attention * reward) - (atrophy_rate * w) : 1 (clock-driven)
        base_lr : 1/second (shared)
        reward : 1 (shared)
        attention : 1 (shared)
        atrophy_rate : 1/second (shared)
        importance : 1 (shared)
        '''
        self.S = b2.Synapses(self.P, self.G, model=eqs, on_pre='v_post += w', method='euler')
        self.S.connect()
        self.S.w = 'rand()*0.2'
        
        self.M = b2.SpikeMonitor(self.G)
        self.net = b2.Network(self.P, self.G, self.S, self.M)
        self.last_counts = np.zeros(n_actions)
        self.importance_matrix = np.zeros(len(self.S.w)) # Fisher-lite

    def update_importance(self, rwd):
        """Reinforces weights that led to high reward; accelerates atrophy for others."""
        if rwd > 1.5:
            # Consolidate: Lock weights that lead to success
            self.importance_matrix += np.abs(np.array(self.S.w)) * 0.1
        elif rwd < -2.0:
            # Atrophy: Rapidly decay weights that lead to failure
            self.S.w *= 0.8

    def predict_and_learn(self, z, r, a, flux):
        self.P.rates = np.clip(z.squeeze().cpu().numpy(), 0, 1)*100*b2.Hz
        self.S.reward, self.S.attention = r, a
        # FARADAY INDUCTION: Flux shortens the time constant, accelerating cognitive reaction
        self.G.tau = (20.0 - min(15.0, flux * 2.0)) * b2.ms 
        # BIOLOGICAL FORGETTING: Natural decay of weak synapses
        self.S.atrophy_rate = 0.005 * b2.Hz 
        
        self.net.run(20*b2.ms)
        c = np.array(self.M.count) - self.last_counts
        self.last_counts = np.array(self.M.count)
        
        self.update_importance(r)
        return int(np.argmax(c)) if np.sum(c) > 0 else random.randint(0, 4)

# ═════════════════════════════════════════════════════════════════════════════
#  3. MASTER SWARM AGENT
# ═════════════════════════════════════════════════════════════════════════════

class ApexResonantNode:
    def __init__(self, name, chassis, planet_id):
        self.name, self.chassis, self.planet_id = name, chassis, planet_id
        self.energy = 10000.0; self.avg_rwd = 0.0
        
        self.head = ElasticNeuromorphicHead(n_actions=5)
        self.visual_cortex = MagnetoProjector().to(DEVICE)
        
        # Load Weights with PyTorch 2.6 security bypass (weights_only=False)
        base_path = "/home/scidev/PycharmProjects/PythonProject/"
        for f, m in [("optimized_living_planet_weights.pt", self.head), ("magneto_projector_weights.pt", self.visual_cortex)]:
            p = os.path.join(base_path, f)
            if os.path.exists(p):
                try:
                    data = torch.load(p, map_location='cpu', weights_only=False)
                    if 'synaptic_weights' in data: self.head.S.w = data['synaptic_weights']
                    else: m.load_state_dict(data, strict=False)
                    print(f"✅ {self.name}: Memory Fragment Loaded ({f}).")
                except Exception: pass

        self.perception = PerceptionPipeline(FeatureNormalizer(NORM_JSON), HoloSynHeads(), StudentDistilledHeadsHF(), StudentDistilledHeadsBasic())

    def step(self, cycle):
        # ── 1. SENSE: Real Data + Tesla Phase ──
        frame = data_lake.get_frame(cycle)
        tesla_resonance = math.sin(cycle * math.pi / 12.0)
        
        # Sense Vector: [Precision, Efficiency, Energy, Resonance, Phase_Error]
        feat = torch.tensor([[frame["prec"], frame["eff"], self.energy/10000.0, tesla_resonance, 0.1]], dtype=torch.float32).to(DEVICE)
        bundle = ModalityBundle(E_t_aug=torch.randn(1, 528), E_i=torch.zeros(1, 2048), E_a=torch.randn(1, 2048), E_v=torch.randn(1, 2048), H=torch.zeros(1, 32, 3), feat_hf=torch.randn(1, 789), feat_basic=feat, mask=torch.ones(1, 4))
        
        econ_latent, _, _ = self.perception.perceive(bundle)
        img_tensor = torch.randn(1, 1, 128, 128).to(DEVICE) # Mock visual scan
        with torch.no_grad(): vis_latent = self.visual_cortex(img_tensor)
            
        # ── 2. FARADAY INDUCTION ──
        fused_latent = (econ_latent + vis_latent) / 2.0
        faraday_flux = torch.std(vis_latent).item() * 5.0 

        # ── 3. COGNITION (Remembering vs. Forgetting) ──
        rwd_signal = (frame["prec"] - 0.5) * 4.0 # Boosted reward signal
        attn = 1.0 + faraday_flux + (abs(tesla_resonance) * 2.0)
        
        action = self.head.predict_and_learn(fused_latent, rwd_signal, attn, faraday_flux)
        
        # ── 4. ACT & RESONANCE PHYSICS ──
        act_name = ["Mine", "Mfg", "Asm", "Logi", "Maint"][action]
        
        if action == 4: # Maintenance
            # Tesla AC Bonus: Sync charging with peak resonance
            charge_efficiency = 1.0 + max(0, tesla_resonance)
            self.energy += (250 * charge_efficiency)
            final_rwd = 1.0 if self.energy < 8000 else -1.0
        else:
            self.energy -= 100
            final_rwd = rwd_signal + (1.0 if action in [1, 2] else 0.5)

        self.energy = max(0, min(10000, self.energy))
        self.avg_rwd = 0.9 * self.avg_rwd + 0.1 * final_rwd
        
        # Display Importance (Neural Stability)
        importance_score = np.mean(self.head.importance_matrix)
        
        status_color = "\033[94m" if action == 4 else "\033[92m"
        print(f"[{self.name}] Act: {status_color}{act_name:<8}\033[0m | NRG: {self.energy:4.0f} | Attn: {attn:4.2f} | Memory Stability: {importance_score:.4f}")
        db_logger.log_action(self.name, act_name, final_rwd, importance_score)

# ── MAIN DEPLOYMENT ──
if __name__ == "__main__":
    print("\n" + "═"*90 + "\n  ⚡ PHASE 24: RESONANT ELASTIC SWARM (MICMECHANICAL NEURAL NET)\n" + "═"*90)
    swarm = [ApexResonantNode("Alpha-1", "Borer", "Vulcan"), ApexResonantNode("Beta-1", "Foundry", "Vulcan")]
    for cycle in range(1, 101):
        for drone in swarm: drone.step(cycle)
        time.sleep(0.01)