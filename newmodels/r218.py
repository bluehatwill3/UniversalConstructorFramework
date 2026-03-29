"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PLANET FACTORY: PHASE 18 - APEX HYBRID TRAINER (FIXED)        ║
║                                                                              ║
║  Optimized for Survival & Distillation. Solves the 'Metabolic Death Spiral'  ║
║  by prioritizing energy recovery reward logic and robust path resolution.    ║
║  Features: Experience Replay, Teacher Distillation, and RPE-Attention.       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import time
import numpy as np
import torch
import torch.nn as nn
import os
import csv
import sqlite3
import random
from collections import deque
from typing import List, Dict, Tuple

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
#  1. HYBRID INTELLIGENCE: THE TEACHER HEAD
# ═════════════════════════════════════════════════════════════════════════════

class PyTorchTeacherHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class ExperienceMemory:
    def __init__(self, capacity=1000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, attention):
        self.memory.append((state, action, reward, attention))

    def sample(self, batch_size=32):
        return random.sample(self.memory, min(len(self.memory), batch_size))


# ═════════════════════════════════════════════════════════════════════════════
#  2. ROBUST DATA INGESTION
# ═════════════════════════════════════════════════════════════════════════════

class DataLoader:
    def __init__(self, csv_name="autocad_data.csv", db_name="collected_data.db"):
        # Aggressive path resolution for diverse local environments
        search_dirs = [
            os.getcwd(),
            os.path.dirname(os.path.abspath(__file__)),
            "/home/scidev/PycharmProjects/PythonProject/",
            os.path.expanduser("~/PycharmProjects/PythonProject/")
        ]

        self.csv_path = csv_name
        self.db_path = db_name

        for d in search_dirs:
            cp = os.path.join(d, csv_name)
            dp = os.path.join(d, db_name)
            if os.path.exists(cp): self.csv_path = cp
            if os.path.exists(dp): self.db_path = dp

        self.autocad_data = self._load_csv()
        self.economy_data = self._load_db()

    def _load_csv(self) -> List[Dict]:
        data = []
        if os.path.exists(self.csv_path):
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
            print(f"📊 AutoCAD Ingested: {len(data)} frames from {self.csv_path}")
        return data

    def _load_db(self) -> List[Dict]:
        data = []
        if os.path.exists(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='planetary_economy';")
                if cursor.fetchone():
                    cursor.execute(
                        "SELECT energy, ore, semi_finished, finished_goods, propellant FROM planetary_economy LIMIT 5000")
                    rows = cursor.fetchall()
                    for r in rows:
                        data.append({"energy": r[0], "ore": r[1], "semi": r[2], "goods": r[3], "prop": r[4]})
                conn.close()
                print(f"💾 Economy DB Connected: {len(data)} rows from {self.db_path}")
            except sqlite3.Error:
                pass
        return data


# ═════════════════════════════════════════════════════════════════════════════
#  3. THE STUDENT SNN
# ═════════════════════════════════════════════════════════════════════════════

class StudentNeuromorphicHead(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.n_actions = n_actions
        if not HAS_NEUROMORPHIC: return

        b2.start_scope()
        self.P = b2.PoissonGroup(256, rates=np.zeros(256) * b2.Hz)
        self.G = b2.NeuronGroup(n_actions, 'dv/dt = (I - v) / (10*ms) : 1\nI : 1\nmood_modifier : 1',
                                threshold='v > (1.0 + mood_modifier)', reset='v=0', refractory=2 * b2.ms,
                                method='euler')

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
        self.S = b2.Synapses(self.P, self.G, model=syn_eqs, on_pre='v_post += w; Apre += 0.01; c += Apost',
                             on_post='Apost -= 0.01; c += Apre', method='euler')
        self.S.connect();
        self.S.w = 'rand() * 0.3'
        self.S.taupre = 20 * b2.ms;
        self.S.taupost = 20 * b2.ms;
        self.S.tau_c = 50 * b2.ms
        self.S.base_lr = 5.0 * b2.Hz;
        self.S.w_decay = 0.01 * b2.Hz;
        self.S.reward = 0.0;
        self.S.attention = 1.0;
        self.G.mood_modifier = 0.0

        self.M = b2.SpikeMonitor(self.G);
        self.net = b2.Network(self.P, self.G, self.S, self.M)
        self.last_counts = np.zeros(n_actions)

    def train_step(self, z: torch.Tensor, reward_signal: float, attention_signal: float) -> tuple:
        if not HAS_NEUROMORPHIC: return 0, np.zeros(self.n_actions)
        self.P.rates = np.clip(z.squeeze().cpu().numpy(), 0, 1) * 100 * b2.Hz
        self.S.reward, self.S.attention = reward_signal, attention_signal
        self.net.run(20 * b2.ms)
        step_counts = np.array(self.M.count) - self.last_counts
        self.last_counts = np.array(self.M.count)
        return (
            int(np.argmax(step_counts)) if np.sum(step_counts) > 0 else np.random.randint(0, self.n_actions)), np.array(
            self.S.w)


# ═════════════════════════════════════════════════════════════════════════════
#  4. APEX TRAINING ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def run_apex_training():
    print("═" * 80)
    print("  🧠 PLANET FACTORY: PHASE 18 - APEX HYBRID TRAINER")
    print("═" * 80)

    # 1. Models & Teacher Setup
    try:
        hf_model = torch.jit.load("student_distilled_heads_hf.torchscript.pt")
        basic_model = torch.jit.load("student_distilled_heads.torchscript.pt")
        print("✅ JIT Edge Models online.")
    except Exception:
        hf_model, basic_model = StudentDistilledHeadsHF(), StudentDistilledHeadsBasic()

    teacher = PyTorchTeacherHead().to(DEVICE)

    # Aggressive search for weights
    t_weights = "robot_arm_snn_head.pt"
    search_dirs = [os.getcwd(), os.path.dirname(os.path.abspath(__file__)),
                   "/home/scidev/PycharmProjects/PythonProject/",
                   os.path.expanduser("~/PycharmProjects/PythonProject/")]

    potential_paths = [t_weights] + [os.path.join(d, t_weights) for d in search_dirs]
    teacher_loaded = False

    for p in potential_paths:
        if os.path.exists(p):
            try:
                teacher.load_state_dict(torch.load(p, map_location=DEVICE), strict=False)
                teacher.eval()
                print(f"✅ Distillation Teacher online ({p}).")
                teacher_loaded = True
                break
            except Exception:
                pass

    if not teacher_loaded: print("⚠️ Teacher offline. Using random exploratory initialization.")

    perception = PerceptionPipeline(FeatureNormalizer(NORM_JSON), HoloSynHeads(), hf_model, basic_model)
    student = StudentNeuromorphicHead(n_actions=3)
    data = DataLoader()
    memory = ExperienceMemory()

    # Metabolic State
    current_energy = 5000.0
    epochs = 300
    avg_reward = 0.0
    best_reward = -999.0
    best_weights = None

    print(f"\n🚀 Commencing Hybrid Knowledge Distillation ({epochs} Epochs)...")

    for epoch in range(1, epochs + 1):
        # ── SENSORY FETCH ──
        if data.autocad_data:
            ac = data.autocad_data[epoch % len(data.autocad_data)]
        else:
            # FIX: Use dynamic random bounds so the model isn't starved of prediction error
            ac = {
                "geo_prec": float(np.random.uniform(0.4, 0.95)),
                "asm_eff": float(np.random.uniform(0.4, 0.95)),
                "prec_chg": float(np.random.uniform(0.0, 0.1))
            }

        current_prec_chg = ac["prec_chg"]

        # Periodic "Precision Shock"
        if epoch % 50 == 0:
            current_prec_chg = 0.8  # FIX: Use a local variable to prevent permanently corrupting the dataset

        feat_basic = torch.tensor([[ac["geo_prec"], ac["asm_eff"], current_energy / 10000.0, 0.1, 0.1]],
                                  dtype=torch.float32).to(DEVICE)
        bundle = ModalityBundle(E_t_aug=torch.randn(1, 528), E_i=torch.zeros(1, 2048), E_a=torch.randn(1, 2048),
                                E_v=torch.randn(1, 2048), H=torch.zeros(1, 32, 3), feat_hf=torch.randn(1, 789),
                                feat_basic=feat_basic, mask=torch.ones(1, 4))

        fused_emb, _, _ = perception.perceive(bundle)

        # ── TEACHER ADVICE ──
        teacher_action = 1  # Default med-load
        if teacher_loaded:
            with torch.no_grad():
                teacher_logits = teacher(fused_emb)
                teacher_action = torch.argmax(teacher_logits, dim=-1).item()

        # ── REWARD & SURVIVAL LOGIC ──
        base_reward = (ac["asm_eff"] - 0.5) * 2.0
        if ac["geo_prec"] > 0.8: base_reward += 1.0

        # Dynamic Attention based on Reward Prediction Error
        reward_error = abs(base_reward - avg_reward)
        attention_signal = 1.0 + (reward_error * 2.5) + (abs(current_prec_chg) * 8.0)
        avg_reward = 0.9 * avg_reward + 0.1 * base_reward

        # ── STUDENT STEP ──
        student_action, weights = student.train_step(fused_emb, base_reward, attention_signal)

        # ── INTEGRATED FINAL REWARD ──
        final_reward = base_reward

        # Action Physics
        energy_loss = 0
        if student_action == 0:
            energy_loss = 120.0
        elif student_action == 1:
            energy_loss = 60.0
        else:  # Action 2: Recovery
            energy_loss = -250.0
            if current_energy < 2000.0: final_reward += 2.0  # ⚡ GRID RECOVERY DOPAMINE SPIKE

        current_energy = max(0, min(10000.0, current_energy - energy_loss))

        # Penalty for bankruptcy EXEMPTING the recovery action
        if current_energy < 1500 and student_action != 2:
            final_reward -= 3.0

        # Distillation Bonus
        if teacher_loaded and student_action == teacher_action:
            final_reward += 0.8

        memory.push(fused_emb, student_action, final_reward, attention_signal)

        # Biological Replay
        if epoch % 10 == 0 and len(memory.memory) > 32:
            batch = memory.sample(16)
            for m_state, m_act, m_rew, m_attn in batch:
                student.train_step(m_state, m_rew, m_attn)

        if final_reward > best_reward:
            best_reward = final_reward
            best_weights = weights.copy()

        if epoch % 25 == 0:
            print(
                f"  [Epoch {epoch:03d}] Act: {student_action} (Teach: {teacher_action}) | NRG: {current_energy:4.0f} | Attn: {attention_signal:.2f} -> Rwd: {final_reward:+.1f}")

    print(f"\n🏁 Training Complete. Best Momentary Reward: {best_reward:.2f}")
    if best_weights is not None:
        torch.save({"synaptic_weights": best_weights}, "optimized_living_planet_weights.pt")
        print("💾 Optimized weights saved.")


if __name__ == "__main__":
    run_apex_training()