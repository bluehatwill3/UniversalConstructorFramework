"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PLANET FACTORY: PHASE 15 - SNN TRAINING ENGINE                ║
║                                                                              ║
║  This engine is dedicated to training the Spiking Neural Network (SNN)       ║
║  for the Living Planet Factory. It ingests historical telemetry (CSV/DB)     ║
║  and pre-trained TorchScript models to optimize interplanetary economics.    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import time
import numpy as np
import torch
import torch.nn as nn
import os
import csv
import sqlite3
from typing import List, Dict

# Import our Brains and Architecture from Phase 1
from ucf import (
    PerceptionPipeline, FeatureNormalizer,
    HoloSynHeads, StudentDistilledHeadsHF, StudentDistilledHeadsBasic,
    ModalityBundle, NORM_JSON, DEVICE, SafetyGovernor, IndustrialThresholds
)

# ── 🧠 NEUROMORPHIC BACKEND ──────────────────────────────────────────────────
try:
    import brian2 as b2
    b2.prefs.codegen.target = 'numpy'
    HAS_NEUROMORPHIC = True
except ImportError:
    HAS_NEUROMORPHIC = False
    print("⚠️ Brian2 not installed. SNN Simulation will be bypassed.")

# ═════════════════════════════════════════════════════════════════════════════
#  1. TRAINING DATA INGESTION
# ═════════════════════════════════════════════════════════════════════════════

class DataLoader:
    """Loads historical AutoCAD and SQLite telemetry to build training epochs."""
    def __init__(self, csv_path="autocad_data.csv", db_path="collected_data.db"):
        self.csv_path = csv_path
        self.db_path = db_path
        self.autocad_data = self._load_csv()
        self.economy_data = self._load_db()

    def _load_csv(self) -> List[Dict]:
        data = []
        if os.path.exists(self.csv_path):
            print(f"📊 Loading engineering telemetry from {self.csv_path}...")
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        data.append({
                            "geo_prec": float(row["Geometric Precision"]),
                            "asm_eff": float(row["Assembly Efficiency"]),
                            "prec_chg": float(row["Precision Change"])
                        })
                    except KeyError:
                        pass
            print(f"✅ Loaded {len(data)} frames of AutoCAD telemetry.")
        else:
            print(f"⚠️ {self.csv_path} not found. Mocking AutoCAD data.")
        return data

    def _load_db(self) -> List[Dict]:
        data = []
        if os.path.exists(self.db_path):
            print(f"💾 Connecting to historical economy DB: {self.db_path}...")
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                # Check if the table exists before querying
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='planetary_economy';")
                if cursor.fetchone():
                    cursor.execute("SELECT energy, ore, semi_finished, finished_goods, propellant FROM planetary_economy LIMIT 1000")
                    rows = cursor.fetchall()
                    for r in rows:
                        data.append({
                            "energy": r[0], "ore": r[1], "semi": r[2], "goods": r[3], "prop": r[4]
                        })
                    print(f"✅ Loaded {len(data)} rows of historical economy data.")
                else:
                    print("⚠️ 'planetary_economy' table not found in DB.")
                conn.close()
            except sqlite3.Error as e:
                print(f"⚠️ SQLite error: {e}")
        else:
            print(f"⚠️ {self.db_path} not found. Mocking Economy data.")
        return data

# ═════════════════════════════════════════════════════════════════════════════
#  2. ADVANCED TRAINING SNN
# ═════════════════════════════════════════════════════════════════════════════

class TrainingNeuromorphicHead(nn.Module):
    """An SNN Action Head optimized for fast-forward training loops."""
    def __init__(self, n_actions: int):
        super().__init__()
        self.n_actions = n_actions
        if not HAS_NEUROMORPHIC: return

        b2.start_scope()
        self.P = b2.PoissonGroup(256, rates=np.zeros(256)*b2.Hz)
        self.G = b2.NeuronGroup(n_actions, 'dv/dt = (I - v) / (10*ms) : 1\nI : 1\nmood_modifier : 1',
                                threshold='v > (1.0 + mood_modifier)', reset='v=0', refractory=2*b2.ms, method='euler')

        # Simplified STDP + Reward equation for training stability
        syn_eqs = '''
        dApre/dt = -Apre / taupre : 1 (event-driven)
        dApost/dt = -Apost / taupost : 1 (event-driven)
        dc/dt = -c / tau_c : 1 (clock-driven)  
        dw/dt = (base_lr) * c * reward - w_decay * w : 1 (clock-driven)
        
        taupre : second (shared)
        taupost : second (shared)
        tau_c : second (shared)
        base_lr : 1/second (shared)
        reward : 1 (shared)
        w_decay : 1/second (shared)
        '''
        self.S = b2.Synapses(self.P, self.G, model=syn_eqs,
                             on_pre='v_post += w; Apre += 0.01; c += Apost',
                             on_post='Apost -= 0.01; c += Apre', method='euler')
        self.S.connect()
        self.S.w = 'rand() * 0.3'

        self.S.taupre = 20*b2.ms; self.S.taupost = 20*b2.ms; self.S.tau_c = 50*b2.ms
        self.S.base_lr = 5.0*b2.Hz; self.S.w_decay = 0.01*b2.Hz
        self.S.reward = 0.0; self.G.mood_modifier = 0.0

        self.M = b2.SpikeMonitor(self.G)
        self.net = b2.Network(self.P, self.G, self.S, self.M)
        self.last_counts = np.zeros(n_actions)

    def train_step(self, z: torch.Tensor, reward_signal: float) -> tuple:
        if not HAS_NEUROMORPHIC: return 0, np.zeros(self.n_actions)

        self.P.rates = np.clip(z.squeeze().cpu().numpy(), 0, 1) * 100 * b2.Hz
        self.S.reward = reward_signal

        self.net.run(20*b2.ms)

        step_counts = np.array(self.M.count) - self.last_counts
        self.last_counts = np.array(self.M.count)

        action = int(np.argmax(step_counts)) if np.sum(step_counts) > 0 else np.random.randint(0, self.n_actions)
        return action, np.array(self.S.w)

# ═════════════════════════════════════════════════════════════════════════════
#  3. TRAINING ENGINE LOOP
# ═════════════════════════════════════════════════════════════════════════════

def run_training_engine():
    print("═" * 80)
    print("  🧠 PLANET FACTORY: MACRO-ECONOMIC SNN TRAINER")
    print("═" * 80)

    # 1. Load Edge-Optimized Models
    print("\n⚡ Loading TorchScript Perception Models...")
    try:
        hf_model = torch.jit.load("student_distilled_heads_hf.torchscript.pt")
        basic_model = torch.jit.load("student_distilled_heads.torchscript.pt")
        print("✅ Edge Models loaded successfully.")
    except Exception as e:
        print(f"⚠️ Failed to load TorchScript models: {e}. Falling back to standard PyTorch.")
        hf_model = StudentDistilledHeadsHF()
        basic_model = StudentDistilledHeadsBasic()

    perception = PerceptionPipeline(FeatureNormalizer(NORM_JSON), HoloSynHeads(), hf_model, basic_model)

    # 2. Initialize SNN
    print("\n🧬 Initializing Training SNN...")
    snn_head = TrainingNeuromorphicHead(n_actions=3)

    # Optional: Start from existing weights
    pretrained_path = "robot_arm_snn_head.pt"
    if os.path.exists(pretrained_path):
        print(f"📥 Loading existing weights from {pretrained_path}...")
        # Since Brian2 weights are numpy arrays, we handle state injection differently
        # For this trainer, we'll assume we are training a fresh set of weights to save later
        print("   -> Starting training from scratch to generate new weights.")

    # 3. Load Historical Data
    data_loader = DataLoader()

    epochs = 100
    print(f"\n🚀 Commencing Training Loop ({epochs} Epochs)...")

    best_reward = -999.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        epoch_reward = 0.0

        # Simulate an economic cycle using data
        # Mocking values if data is missing for demonstration
        geo_prec = data_loader.autocad_data[epoch % len(data_loader.autocad_data)]["geo_prec"] if data_loader.autocad_data else np.random.rand()
        energy = data_loader.economy_data[epoch % len(data_loader.economy_data)]["energy"] if data_loader.economy_data else np.random.rand() * 5000

        # Build sensory tensor: [Geo_Prec, Energy_Norm, Mock, Mock, Mock]
        feat_array = [[geo_prec, energy / 10000.0, np.random.rand(), np.random.rand(), np.random.rand()]]
        feat_basic_tensor = torch.tensor(feat_array, dtype=torch.float32).to(DEVICE)

        bundle = ModalityBundle(
            E_t_aug=torch.randn(1, 528),
            E_i=torch.zeros(1, 2048),
            E_a=torch.randn(1, 2048),
            E_v=torch.randn(1, 2048),
            H=torch.zeros(1, 32, 3),
            feat_hf=torch.randn(1, 789),
            feat_basic=feat_basic_tensor,
            mask=torch.ones(1, 4),
        )

        # Forward Pass
        fused_emb, _, _ = perception.perceive(bundle)

        # Determine Training Reward based on data state
        # Example Logic: High geometric precision and high energy yields a positive reward
        if geo_prec > 0.5 and energy > 2000.0:
            step_reward = 2.0
        elif energy < 1000.0:
            step_reward = -1.0 # Scarcity penalty
        else:
            step_reward = 0.1

        epoch_reward += step_reward

        # SNN Learning Step
        action, current_weights = snn_head.train_step(fused_emb, step_reward)

        # Save best weights
        if epoch_reward > best_reward:
            best_reward = epoch_reward
            best_weights = current_weights.copy()

        if epoch % 10 == 0:
            print(f"  [Epoch {epoch:03d}] Action: {action} | GeoPrec: {geo_prec:.2f} | Energy: {energy:4.0f} -> Reward: {step_reward:+.1f}")

    print("\n🏁 Training Complete.")
    print(f"🏆 Best Epoch Reward: {best_reward:.1f}")

    # Save the newly trained weights
    save_path = "optimized_living_planet_weights.pt"
    # Note: Brian2 weights are typically saved as numpy arrays, not torch state_dicts.
    # We save it in a format that could be loaded back into the SNN.
    if best_weights is not None:
        torch.save({"synaptic_weights": best_weights}, save_path)
        print(f"💾 Saved optimized SNN weights to {save_path}")

if __name__ == "__main__":
    run_training_engine()