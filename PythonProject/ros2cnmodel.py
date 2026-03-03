"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PLANET FACTORY: PHASE 16 - DEEP LEARNING OPTIMIZATION         ║
║                                                                              ║
║  Advanced SNN Training Engine. Features Multi-Objective Reward Functions,    ║
║  AutoCAD-driven Attention Scaling (Acetylcholine), and robust data parsing   ║
║  to perfect the Spiking Neural Network for Edge Deployment.                  ║
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
    def __init__(self, csv_name="autocad_data.csv", db_name="collected_data.db"):
        # Resolve absolute paths to ensure files are found regardless of execution directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(base_dir, csv_name)
        self.db_path = os.path.join(base_dir, db_name)

        # Fallback to local CWD if not in script dir
        if not os.path.exists(self.csv_path): self.csv_path = csv_name
        if not os.path.exists(self.db_path): self.db_path = db_name

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
                            "geo_prec": float(row.get("Geometric Precision", 0.5)),
                            "asm_eff": float(row.get("Assembly Efficiency", 0.5)),
                            "prec_chg": float(row.get("Precision Change", 0.0))
                        })
                    except (KeyError, ValueError):
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
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='planetary_economy';")
                if cursor.fetchone():
                    cursor.execute("SELECT energy, ore, semi_finished, finished_goods, propellant FROM planetary_economy LIMIT 5000")
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

        # Phase 16: Re-introduced Attention (Acetylcholine) for data-driven learning spikes
        syn_eqs = '''
        dApre/dt = -Apre / taupre : 1 (event-driven)
        dApost/dt = -Apost / taupost : 1 (event-driven)
        dc/dt = -c / tau_c : 1 (clock-driven)  
        dw/dt = (base_lr * attention) * c * reward - w_decay * w : 1 (clock-driven)
        
        taupre : second (shared)
        taupost : second (shared)
        tau_c : second (shared)
        base_lr : 1/second (shared)
        reward : 1 (shared)
        attention : 1 (shared)
        w_decay : 1/second (shared)
        '''
        self.S = b2.Synapses(self.P, self.G, model=syn_eqs,
                             on_pre='v_post += w; Apre += 0.01; c += Apost',
                             on_post='Apost -= 0.01; c += Apre', method='euler')
        self.S.connect()
        self.S.w = 'rand() * 0.3'

        self.S.taupre = 20*b2.ms; self.S.taupost = 20*b2.ms; self.S.tau_c = 50*b2.ms
        self.S.base_lr = 5.0*b2.Hz; self.S.w_decay = 0.01*b2.Hz
        self.S.reward = 0.0; self.S.attention = 1.0; self.G.mood_modifier = 0.0

        self.M = b2.SpikeMonitor(self.G)
        self.net = b2.Network(self.P, self.G, self.S, self.M)
        self.last_counts = np.zeros(n_actions)

    def train_step(self, z: torch.Tensor, reward_signal: float, attention_signal: float) -> tuple:
        if not HAS_NEUROMORPHIC: return 0, np.zeros(self.n_actions)

        self.P.rates = np.clip(z.squeeze().cpu().numpy(), 0, 1) * 100 * b2.Hz
        self.S.reward = reward_signal
        self.S.attention = attention_signal

        self.net.run(20*b2.ms)

        step_counts = np.array(self.M.count) - self.last_counts
        self.last_counts = np.array(self.M.count)

        action = int(np.argmax(step_counts)) if np.sum(step_counts) > 0 else np.random.randint(0, self.n_actions)
        return action, np.array(self.S.w)

# ═════════════════════════════════════════════════════════════════════════════
#  3. TRAINING ENGINE LOOP (Multi-Objective)
# ═════════════════════════════════════════════════════════════════════════════

def run_training_engine():
    print("═" * 80)
    print("  🧠 PLANET FACTORY: PHASE 16 - MULTI-OBJECTIVE SNN TRAINER")
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

    # 3. Load Historical Data
    data_loader = DataLoader()

    epochs = 200 # Increased for deeper learning
    print(f"\n🚀 Commencing Multi-Objective Training Loop ({epochs} Epochs)...")

    best_reward = -999.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        epoch_reward = 0.0

        # ── FETCH DATA STATE ──
        if data_loader.autocad_data:
            ac_row = data_loader.autocad_data[epoch % len(data_loader.autocad_data)]
        else:
            ac_row = {"geo_prec": np.random.rand(), "asm_eff": np.random.rand(), "prec_chg": 0.0}

        if data_loader.economy_data:
            econ_row = data_loader.economy_data[epoch % len(data_loader.economy_data)]
        else:
            econ_row = {"energy": np.random.rand() * 5000, "ore": 100, "goods": 5}

        # ── SENSORY ENCODING ──
        feat_array = [[ac_row["geo_prec"], ac_row["asm_eff"], econ_row["energy"] / 10000.0, econ_row["ore"] / 1000.0, econ_row["goods"] / 100.0]]
        feat_basic_tensor = torch.tensor(feat_array, dtype=torch.float32).to(DEVICE)

        bundle = ModalityBundle(
            E_t_aug=torch.randn(1, 528), E_i=torch.zeros(1, 2048),
            E_a=torch.randn(1, 2048), E_v=torch.randn(1, 2048),
            H=torch.zeros(1, 32, 3), feat_hf=torch.randn(1, 789),
            feat_basic=feat_basic_tensor, mask=torch.ones(1, 4)
        )

        fused_emb, _, _ = perception.perceive(bundle)

        # ── MULTI-OBJECTIVE REWARD CALCULATION ──
        # 1. Base efficiency reward
        step_reward = (ac_row["asm_eff"] - 0.5) * 2.0
        # 2. Geometric precision bonus
        if ac_row["geo_prec"] > 0.75: step_reward += 1.0
        # 3. Macro-economic penalty (Grid collapse)
        if econ_row["energy"] < 1000.0: step_reward -= 2.0

        epoch_reward += step_reward

        # ── BIOLOGICAL ATTENTION SCALING ──
        # If AutoCAD tolerance is shifting rapidly, spike Acetylcholine (Attention) to learn faster
        attention_signal = 1.0 + abs(ac_row["prec_chg"]) * 5.0

        # SNN Learning Step
        action, current_weights = snn_head.train_step(fused_emb, step_reward, attention_signal)

        if epoch_reward > best_reward:
            best_reward = epoch_reward
            best_weights = current_weights.copy()

        if epoch % 20 == 0:
            print(f"  [Epoch {epoch:03d}] Act: {action} | Geo: {ac_row['geo_prec']:.2f} | NRG: {econ_row['energy']:4.0f} | Attn: {attention_signal:.1f} -> Rwd: {step_reward:+.1f}")

    print("\n🏁 Training Complete.")
    print(f"🏆 Best Epoch Reward Achieved: {best_reward:.2f}")

    # Save the newly trained weights
    save_path = "optimized_living_planet_weights.pt"
    if best_weights is not None:
        torch.save({"synaptic_weights": best_weights}, save_path)
        print(f"💾 Saved optimized SNN weights to {save_path}")

if __name__ == "__main__":
    run_training_engine()