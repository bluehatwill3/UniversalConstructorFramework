"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PLANET FACTORY: PHASE 28 - FARADAY OPTICS SWARM               ║
║                                                                              ║
║  Automated Lenscrafting via Contextual Validation:                           ║
║  - Open-Source CV Data: Ingests real-world industrial textures/defects.      ║
║  - Faraday's Method: Cross-references applied magnetic fields with visual    ║
║    polarization shifts to validate heavy glass integrity.                    ║
║  - Atomic Learning: Validated contexts accelerate Babbage Carry registers    ║
║    to write successful lenscrafting techniques into hardware ROM.            ║
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
    def __init__(self, db_path="faraday_optics.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_db()
    def _init_db(self):
        self.cursor.execute('CREATE TABLE IF NOT EXISTS optic_economy (lens_id TEXT, nrg REAL, purity REAL, precision REAL)')
        self.cursor.execute('CREATE TABLE IF NOT EXISTS lenscraft_actions (crafter_id TEXT, action TEXT, rwd REAL, stability REAL)')
        self.conn.commit()
    def log_action(self, c_id, act, rwd, stability):
        self.cursor.execute('INSERT INTO lenscraft_actions VALUES (?, ?, ?, ?)', (c_id, act, rwd, stability))
        self.conn.commit()

db_logger = DatabaseLogger()

# ── 📊 REAL-WORLD LENS DATA LAKE ─────────────────────────────────────────────
class LensTolerancesLake:
    """Ingests physical constraints for borosilicate of lead (heavy glass)."""
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
        print(f"📊 Lens Tolerances: Ingested {len(self.data)} real-world physical frames.")

    def get_frame(self, cycle):
        if not self.data: return {"prec": float(np.random.uniform(0.6, 0.99)), "eff": float(np.random.uniform(0.5, 0.9))}
        return self.data[cycle % len(self.data)]

lens_lake = LensTolerancesLake()

# ═════════════════════════════════════════════════════════════════════════════
#  1. FARADAY OPTICS VISUAL CORTEX (Polarization & Aberration)
# ═════════════════════════════════════════════════════════════════════════════

class FaradayOpticsLake:
    """
    Simulates Faraday's magneto-optic experiments.
    Fetches real surface texture data to observe polarization shifts under magnetism.
    """
    def __init__(self):
        self.use_synthetic_fallback = True
        print("👁️ Faraday Optics CV: Initializing Public Texture Datasets...")
        if HAS_TORCHVISION:
            try:
                # Using standard open-source datasets to represent raw material surfaces
                transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
                self.dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
                self.use_synthetic_fallback = False
                print(f"✅ Open-Source Texture Dataset Loaded: {len(self.dataset)} material samples.")
            except Exception as e:
                print(f"⚠️ Public CV fetch failed ({e}). Simulating heavy glass aberrations.")

    def scan_lens_material(self, cycle, is_magnetized):
        is_fractured = np.random.rand() < 0.15 # 15% chance of internal micro-fracture
        
        if not self.use_synthetic_fallback:
            img_tensor, label = self.dataset[cycle % len(self.dataset)]
            if is_fractured: img_tensor[:, 50:78, 50:78] = 0.0 # Simulate light scattering defect
            img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        else:
            noise = np.random.randn(128, 128).astype(np.float32)
            if HAS_CV2:
                sigma = np.random.uniform(0.5, 1.5) # Glass smoothness
                domain = cv2.GaussianBlur(noise, (0, 0), sigma)
                _, glass = cv2.threshold(domain, 0, 1, cv2.THRESH_BINARY)
                if is_fractured: cv2.line(glass, (30, 30), (90, 90), 0, 3) # Internal crack
                
                # Apply Faraday rotation mask if magnetized
                if is_magnetized:
                    M = cv2.getRotationMatrix2D((64, 64), 45, 1.0) # Rotate polarization plane
                    glass = cv2.warpAffine(glass, M, (128, 128))
                    
                glass = cv2.GaussianBlur((glass * 255).astype(np.uint8), (3, 3), 0)
                optic_img = np.clip(glass.astype(np.float32) + np.random.normal(0, 5, (128, 128)), 0, 255) / 255.0
                img_tensor = torch.from_numpy(optic_img).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            else:
                img_tensor = torch.randn(1, 1, 128, 128).to(DEVICE)
                if is_fractured: img_tensor[:, :, 40:80, 40:80] = -1.0
                
        # Calculate the variance of the light passing through (Polarization Shift)
        polarization_shift = torch.var(img_tensor).item() * 15.0
        return img_tensor, is_fractured, polarization_shift

cv_lake = FaradayOpticsLake()

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
#  2. BABBAGE-WILSON ATOMIC HEAD (Context Validation Logic)
# ═════════════════════════════════════════════════════════════════════════════

class AnalyticalAtomicHead:
    """SNN utilizing Babbage's Carry logic and Wilson's RISC Write-Back."""
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
        """Faraday Accelerator: Validated scientific contexts drive rapid memory stability."""
        carry_multiplier = 3.0 if context_validated else 1.0
        
        if rwd > 0.5:
            self.success_registers[action_id] += (0.1 * carry_multiplier)
            if self.success_registers[action_id] > 1.0:
                target_synapses = self.S.stability[self.S.j == action_id]
                self.S.stability[self.S.j == action_id] = np.clip(target_synapses + 0.05, 0, 0.99)
                self.success_registers[action_id] = 0.0 
        elif rwd < -2.0:
            fault_multiplier = 2.0 if not context_validated else 1.0
            self.S.stability[self.S.j == action_id] *= (0.7 / fault_multiplier)
            self.S.w[self.S.j == action_id] *= 0.8
            self.success_registers[action_id] = 0.0

    def predict(self, z, flux):
        if not HAS_NEUROMORPHIC: return random.randint(0, 4)
        self.P.rates = np.clip(z.squeeze().cpu().numpy(), 0, 1)*100*b2.Hz
        self.G.tau = (20.0 - min(15.0, flux * 1.5)) * b2.ms 
        self.net.run(20*b2.ms)
        c = np.array(self.M.count) - self.last_counts
        self.last_counts = np.array(self.M.count)
        return int(np.argmax(c)) if np.sum(c) > 0 else random.randint(0, 4)

    def learn(self, rwd, attn):
        self.S.reward, self.S.attention = rwd, attn
        self.S.atrophy_rate = 0.002 * b2.Hz 

class PyTorchTeacherHead(nn.Module):
    def __init__(self, out_dim=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, out_dim))
    def forward(self, x): return self.net(x)

# ═════════════════════════════════════════════════════════════════════════════
#  3. AUTOMATED LENSCRAFTER AGENT
# ═════════════════════════════════════════════════════════════════════════════

class FaradayLenscrafter:
    def __init__(self, name, chassis, planet_id):
        self.name, self.chassis, self.planet_id = name, chassis, planet_id
        self.energy = 10000.0; self.avg_rwd = 0.0
        
        self.head = AnalyticalAtomicHead(n_actions=5)
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
                except Exception: pass

        self.has_teacher = False
        t_name = "robot_arm_snn_head.pt"
        p = os.path.join(base_path, t_name)
        if os.path.exists(p):
            try:
                checkpoint = torch.load(p, map_location=DEVICE, weights_only=False)
                out_dim = checkpoint['net.2.bias'].shape[0] if 'net.2.bias' in checkpoint else 5
                self.teacher = PyTorchTeacherHead(out_dim=out_dim).to(DEVICE)
                self.teacher.load_state_dict(checkpoint, strict=False)
                self.teacher.eval(); self.has_teacher = True
                self.teacher_out_dim = out_dim
            except Exception: pass

        self.perception = PerceptionPipeline(FeatureNormalizer(NORM_JSON), HoloSynHeads(), StudentDistilledHeadsHF(), StudentDistilledHeadsBasic())

    def step(self, cycle):
        # ── 1. MECHANICAL & AC SENSE ──
        frame = lens_lake.get_frame(cycle)
        tesla_resonance = math.sin(cycle * math.pi / 12.0)
        
        feat = torch.tensor([[frame["prec"], frame["eff"], self.energy/10000.0, tesla_resonance, 0.1]], dtype=torch.float32).to(DEVICE)
        bundle = ModalityBundle(E_t_aug=torch.randn(1, 528), E_i=torch.zeros(1, 2048), E_a=torch.randn(1, 2048), E_v=torch.randn(1, 2048), H=torch.zeros(1, 32, 3), feat_hf=torch.randn(1, 789), feat_basic=feat, mask=torch.ones(1, 4))
        
        econ_latent, _, _ = self.perception.perceive(bundle)

        # ── 2. COGNITIVE PREDICTION ──
        # Predict action blindly first to see if AI wants to apply magnetism
        action = self.head.predict(econ_latent, flux=0.1)
        
        # Override with Distillation occasionally
        if self.has_teacher and random.random() < 0.1:
            with torch.no_grad():
                t_logits = self.teacher(econ_latent)
                padded = torch.full((t_logits.shape[0], 5), -float('inf')).to(DEVICE)
                padded[:, :self.teacher_out_dim] = t_logits
                action = torch.argmax(padded, dim=-1).item()

        # ── 3. LENSCRAFTING ACTIONS ──
        act_name = ["Pour", "Anneal", "Polish", "Magnetize", "Inspect"][action]
        is_magnetized = (action == 3)
        
        # Now scan the optical material
        img_tensor, is_fractured, pol_shift = cv_lake.scan_lens_material(cycle, is_magnetized)
        with torch.no_grad(): vis_latent = self.visual_cortex(img_tensor)
            
        fused_latent = (econ_latent + vis_latent) / 2.0
        faraday_flux = torch.std(vis_latent).item() * 5.0 

        # ── 4. FARADAY'S CONTEXT VALIDATION ──
        # Scientific Method Validation: 
        # If magnetized, we EXPECT high polarization shift. 
        # If fractured, light scatters erratically, invalidating physics models.
        expected_shift = (action == 3)
        actual_shift_high = pol_shift > 1.5
        
        if is_fractured:
            context_validated = False # Fracture breaks the expected physics
        else:
            context_validated = (expected_shift == actual_shift_high)

        # ── 5. PHYSICS & REWARD PIPELINE ──
        rwd_signal = (frame["prec"] - 0.5) * 4.0 
        
        if action == 4: # Inspect/Rest (Energy Recovery)
            ac_boost = max(0.0, tesla_resonance)
            self.energy += (250 * (1.0 + ac_boost))
            final_rwd = 2.0 if self.energy < 8500 else -2.0 
        else:
            if action == 0: self.energy -= 150 # Pour Glass
            elif action == 1: self.energy -= 100 # Anneal
            elif action == 2: self.energy -= 80 # Polish
            elif action == 3: self.energy -= 120 # Magnetize
            
            if is_fractured and action in [2, 3]:
                final_rwd = -5.0
                act_name += " ❌(SHATTER)"
            elif action == 3 and context_validated:
                final_rwd = 3.0 # Faraday Breakthrough Bonus
            else:
                final_rwd = rwd_signal + 1.0

        self.energy = max(0, min(10000, self.energy))
        
        # ── 6. DYNAMIC ATTENTION & WRITE-BACK ──
        val_multiplier = 1.0 if context_validated else 3.0 # Spike attention if physics breaks
        attn = 1.0 + faraday_flux + (abs(final_rwd - self.avg_rwd) * val_multiplier)
        self.avg_rwd = 0.9 * self.avg_rwd + 0.1 * final_rwd

        self.head.learn(final_rwd, attn)
        self.head.write_back_outcome(action, final_rwd, context_validated)
        
        stability_index = np.mean(self.head.S.stability) if HAS_NEUROMORPHIC else 0.0
        
        val_icon = "✔️ " if context_validated else "❓ "
        status_color = "\033[91m" if "SHATTER" in act_name else "\033[94m" if action == 4 else "\033[92m"
        print(f"[{self.name}] {val_icon}Act: {status_color}{act_name:<16}\033[0m | NRG: {self.energy:4.0f} | Pol-Shift: {pol_shift:.2f} | ROM: {stability_index:.4f}")
        db_logger.log_action(self.name, act_name, final_rwd, stability_index)

# ── MAIN DEPLOYMENT ──
if __name__ == "__main__":
    print("\n" + "═"*100 + "\n  🔬 PHASE 28: FARADAY OPTICS - AUTOMATED LENSCRAFTING\n" + "═"*100)
    swarm = [FaradayLenscrafter("Furnace-1", "Borer", "Vulcan"), FaradayLenscrafter("Polisher-1", "Assembler", "Vulcan")]
    for cycle in range(1, 101):
        for drone in swarm: drone.step(cycle)
        time.sleep(0.02)