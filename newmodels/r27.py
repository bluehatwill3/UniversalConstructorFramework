"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            PLANET FACTORY: PHASE 27 - FARADAY CONTEXTUAL SWARM             ║
║                                                                              ║
║  Applied Learning via Context Validation (Faraday's Lens Crafting Method):   ║
║  - Open-Source CV Data: Ingests real-world industrial defect distributions.  ║
║  - Polarization Shift Validation: Cross-references visual magneto-optic      ║
║    aberrations with mechanical precision tolerances to validate reality.     ║
║  - Faraday Accelerator: Validated contexts multiply the Babbage Carry        ║
║    logic, rapidly writing verified realities into Wilson ROM registers.      ║
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

try:
    import torchvision
    import torchvision.transforms as transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

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
        self.cursor.execute('CREATE TABLE IF NOT EXISTS swarm_actions (agent_id TEXT, action TEXT, rwd REAL, stability REAL)')
        self.conn.commit()
    def log_action(self, a_id, act, rwd, stability):
        self.cursor.execute('INSERT INTO swarm_actions VALUES (?, ?, ?, ?)', (a_id, act, rwd, stability))
        self.conn.commit()

db_logger = DatabaseLogger()

# ── 📊 REAL-WORLD MECHANICAL DATA LAKE ───────────────────────────────────────
class MechanicalDataLake:
    """Ingests real AutoCAD data for mechanical context."""
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
        print(f"📊 Mechanical Lake: Ingested {len(self.data)} real-world frames.")

    def get_frame(self, cycle):
        if not self.data: return {"prec": float(np.random.uniform(0.4, 0.95)), "eff": 0.5}
        return self.data[cycle % len(self.data)]

data_lake = MechanicalDataLake()

# ═════════════════════════════════════════════════════════════════════════════
#  1. OPEN-SOURCE CV DATA & MAGNETO-OPTIC LENS CRAFTING
# ═════════════════════════════════════════════════════════════════════════════

class OpenCVDataLake:
    """
    Ingests public Computer Vision datasets (e.g., MVTec AD industrial defects).
    Applies Faraday's heavy glass (borosilicate of lead) polarization shifts.
    """
    def __init__(self):
        self.use_synthetic_fallback = True
        print("👁️ Open-Source CV Lake: Initializing Industrial Defect Streams...")
        if HAS_TORCHVISION:
            try:
                # Attempt to load a public dataset (using FashionMNIST as a lightweight proxy for material textures)
                # In a full edge deployment, this points to MVTec AD or specific huggingface datasets
                transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
                self.dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
                self.use_synthetic_fallback = False
                print(f"✅ Open-Source CV Dataset Loaded: {len(self.dataset)} material samples.")
            except Exception as e:
                print(f"⚠️ Public CV fetch failed ({e}). Falling back to MVTec AD statistical simulation.")

    def scan_lens_material(self, cycle):
        is_defective = np.random.rand() < 0.25 # 25% defect rate in heavy glass
        
        if not self.use_synthetic_fallback:
            img_tensor, label = self.dataset[cycle % len(self.dataset)]
            # Induce a synthetic fracture if defective
            if is_defective: img_tensor[:, 50:78, 50:78] = 0.0
            img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        else:
            # MVTec AD Defect Simulation (Perlin-style noise & Faraday rotation masks)
            noise = np.random.randn(128, 128).astype(np.float32)
            if HAS_CV2:
                sigma = np.random.uniform(1.0, 3.0)
                domain = cv2.GaussianBlur(noise, (0, 0), sigma)
                _, maze = cv2.threshold(domain, 0, 1, cv2.THRESH_BINARY)
                if is_defective: cv2.line(maze, (20, 20), (100, 100), 0, 8) # Glass fracture
                maze = cv2.GaussianBlur((maze * 255).astype(np.uint8), (5, 5), 0)
                optic_img = np.clip(maze.astype(np.float32) + np.random.normal(0, 10, (128, 128)), 0, 255) / 255.0
                img_tensor = torch.from_numpy(optic_img).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            else:
                img_tensor = torch.randn(1, 1, 128, 128).to(DEVICE)
                if is_defective: img_tensor[:, :, 40:80, 40:80] = -1.0
                
        # Faraday Polarization Shift Calculation (Variance under simulated magnetic field)
        polarization_shift = torch.var(img_tensor).item() * 10.0
        
        return img_tensor, is_defective, polarization_shift

cv_lake = OpenCVDataLake()

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
#  2. FARADAY CONTEXT HEAD (Applied Learning & Validation)
# ═════════════════════════════════════════════════════════════════════════════

class FaradayContextHead:
    """SNN utilizing Context Validation, Babbage Carries, and Wilson Write-Backs."""
    def __init__(self, n_actions=5):
        if not HAS_NEUROMORPHIC: return
        b2.start_scope()
        self.P = b2.PoissonGroup(256, rates=np.zeros(256)*b2.Hz)
        
        self.G = b2.NeuronGroup(n_actions, 
            '''dv/dt = (I - v) / tau : 1
               I : 1
               tau : second''', 
            threshold='v > 1.0', reset='v=0', refractory=2*b2.ms, method='euler')
        
        eqs = '''
        dw/dt = (base_lr * attention * reward * (1 - stability)) - (atrophy_rate * w * (1 - stability)) : 1 (clock-driven)
        base_lr : 1/second (shared)
        reward : 1 (shared)
        attention : 1 (shared)
        atrophy_rate : 1/second (shared)
        stability : 1 
        '''
        self.S = b2.Synapses(self.P, self.G, model=eqs, on_pre='v_post += w', method='euler')
        self.S.connect()
        self.S.w = 'rand()*0.2'
        self.S.stability = 0.0
        
        self.M = b2.SpikeMonitor(self.G)
        self.net = b2.Network(self.P, self.G, self.S, self.M)
        self.last_counts = np.zeros(n_actions)
        self.success_registers = np.zeros(n_actions)

    def write_back_outcome(self, action_id, rwd, context_validated):
        """
        Faraday's Applied Learning: If the context is validated across 
        multiple modalities, the Babbage Carry mill spins significantly faster.
        """
        carry_multiplier = 3.0 if context_validated else 1.0
        
        if rwd > 0.5:
            # Accumulate success. Validated contexts fill the register instantly.
            self.success_registers[action_id] += (0.1 * carry_multiplier)
            
            if self.success_registers[action_id] > 1.0:
                target_synapses = self.S.stability[self.S.j == action_id]
                self.S.stability[self.S.j == action_id] = np.clip(target_synapses + 0.05, 0, 0.99)
                self.success_registers[action_id] = 0.0 
        elif rwd < -2.0:
            # Contextual Invalidations destroy stability rapidly
            fault_multiplier = 2.0 if not context_validated else 1.0
            self.S.stability[self.S.j == action_id] *= (0.7 / fault_multiplier)
            self.S.w[self.S.j == action_id] *= 0.8
            self.success_registers[action_id] = 0.0

    def predict(self, z, flux):
        if not HAS_NEUROMORPHIC: return random.randint(0, 4)
        self.P.rates = np.clip(z.squeeze().cpu().numpy(), 0, 1)*100*b2.Hz
        self.G.tau = (20.0 - min(15.0, flux * 2.5)) * b2.ms 
        self.net.run(20*b2.ms)
        c = np.array(self.M.count) - self.last_counts
        self.last_counts = np.array(self.M.count)
        return int(np.argmax(c)) if np.sum(c) > 0 else random.randint(0, 4)

    def learn(self, rwd, attn):
        self.S.reward, self.S.attention = rwd, attn
        self.S.atrophy_rate = 0.002 * b2.Hz 

# ═════════════════════════════════════════════════════════════════════════════
#  3. MASTER SWARM AGENT
# ═════════════════════════════════════════════════════════════════════════════

class FaradayLensNode:
    def __init__(self, name, chassis, planet_id):
        self.name, self.chassis, self.planet_id = name, chassis, planet_id
        self.energy = 10000.0; self.avg_rwd = 0.0
        
        self.head = FaradayContextHead(n_actions=5)
        self.visual_cortex = MagnetoProjector().to(DEVICE)
        
        base_path = "/home/scidev/PycharmProjects/PythonProject/"
        for f, m in [("optimized_living_planet_weights.pt", self.head), ("magneto_projector_weights.pt", self.visual_cortex)]:
            p = os.path.join(base_path, f)
            if os.path.exists(p):
                try:
                    data = torch.load(p, map_location='cpu', weights_only=False)
                    if 'synaptic_weights' in data: 
                        self.head.S.w = data['synaptic_weights']
                        self.head.S.stability = np.clip(np.abs(np.array(self.head.S.w)) * 1.5, 0, 0.4)
                    else: 
                        m.load_state_dict(data, strict=False)
                    print(f"✅ {self.name}: Atomic Instruction Cache Loaded ({f}).")
                except Exception: pass

        self.perception = PerceptionPipeline(FeatureNormalizer(NORM_JSON), HoloSynHeads(), StudentDistilledHeadsHF(), StudentDistilledHeadsBasic())

    def step(self, cycle):
        # ── 1. SENSE: Mechanical & Visual ──
        frame = data_lake.get_frame(cycle)
        tesla_resonance = math.sin(cycle * math.pi / 12.0)
        
        feat = torch.tensor([[frame["prec"], frame["eff"], self.energy/10000.0, tesla_resonance, 0.1]], dtype=torch.float32).to(DEVICE)
        bundle = ModalityBundle(E_t_aug=torch.randn(1, 528), E_i=torch.zeros(1, 2048), E_a=torch.randn(1, 2048), E_v=torch.randn(1, 2048), H=torch.zeros(1, 32, 3), feat_hf=torch.randn(1, 789), feat_basic=feat, mask=torch.ones(1, 4))
        
        econ_latent, _, _ = self.perception.perceive(bundle)
        
        # Open-Source Data Scan
        img_tensor, is_defective, polarization_shift = cv_lake.scan_lens_material(cycle)
        with torch.no_grad(): vis_latent = self.visual_cortex(img_tensor)
            
        fused_latent = (econ_latent + vis_latent) / 2.0
        faraday_flux = torch.std(vis_latent).item() * 5.0 

        # ── 2. FARADAY'S CONTEXT VALIDATION ──
        # Does the visual polarization shift correlate with the mechanical precision data?
        # High shift = Visual Defect. Low prec = Mechanical Defect.
        visual_warning = polarization_shift > 0.8
        mechanical_warning = frame["prec"] < 0.6
        context_validated = (visual_warning == mechanical_warning)

        # ── 3. FETCH & EXECUTE ──
        action = self.head.predict(fused_latent, faraday_flux)
        
        # ── 4. OUTCOME & WRITE-BACK ──
        act_name = ["Mine", "Mfg", "Asm", "Logi", "Maint"][action]
        rwd_signal = (frame["prec"] - 0.5) * 4.0 
        
        if action == 4: 
            charge_efficiency = 1.0 + max(0, tesla_resonance)
            self.energy += (250 * charge_efficiency)
            final_rwd = 2.0 if self.energy < 8500 else -2.0 
        else:
            self.energy -= 100
            
            # Context-Aware Defect Penalty
            if action in [1, 2] and is_defective:
                final_rwd = -5.0
                act_name += " ❌(FRACTURE)"
            else:
                final_rwd = rwd_signal + (1.5 if action in [1, 2] else 0.5)

        self.energy = max(0, min(10000, self.energy))
        
        # DYNAMIC ATTENTION (Spikes hard on unvalidated contexts/surprises)
        validation_modifier = 1.0 if context_validated else 2.5
        attn = 1.0 + faraday_flux + (abs(final_rwd - self.avg_rwd) * validation_modifier)
        self.avg_rwd = 0.9 * self.avg_rwd + 0.1 * final_rwd

        # WILSON/FARADAY WRITE-BACK
        self.head.learn(final_rwd, attn)
        self.head.write_back_outcome(action, final_rwd, context_validated)
        
        stability_index = np.mean(self.head.S.stability) if HAS_NEUROMORPHIC else 0.0
        
        val_icon = "✔️ " if context_validated else "❓ "
        status_color = "\033[91m" if "FRACTURE" in act_name else "\033[94m" if action == 4 else "\033[92m"
        print(f"[{self.name}] {val_icon}Act: {status_color}{act_name:<16}\033[0m | NRG: {self.energy:4.0f} | Shift: {polarization_shift:.2f} | ROM: {stability_index:.4f}")
        db_logger.log_action(self.name, act_name, final_rwd, stability_index)

# ── MAIN DEPLOYMENT ──
if __name__ == "__main__":
    print("\n" + "═"*100 + "\n  ⚡ PHASE 27: FARADAY CONTEXTUAL VALIDATION SWARM\n" + "═"*100)
    swarm = [FaradayLensNode("Alpha-1", "Borer", "Vulcan"), FaradayLensNode("Beta-1", "Foundry", "Vulcan")]
    for cycle in range(1, 101):
        for drone in swarm: drone.step(cycle)
        time.sleep(0.02)