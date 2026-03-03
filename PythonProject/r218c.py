"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PLANET FACTORY: PHASE 18 - APEX HYBRID TRAINER                ║
║                                                                              ║
║  The definitive SNN Training Engine. Features Experience Replay Buffers,     ║
║  Teacher-Student Knowledge Distillation from mature .pt models, and          ║
║  dynamic Acetylcholine (Attention) scaling based on Reward Prediction Error. ║
║  Remodeled for pure high-throughput data-driven optimization.                ║
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
    """Loads pre-trained .pt weights to guide the Student SNN during training."""

    def __init__(self, input_dim=256, hidden_dim=64, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ═════════════════════════════════════════════════════════════════════════════
#  2. BIOLOGICAL EXPERIENCE REPLAY
# ═════════════════════════════════════════════════════════════════════════════

class ExperienceMemory:
    """Consolidates high-impact memories to prevent catastrophic forgetting."""

    def __init__(self, capacity=1000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, attention):
        self.memory.append((state, action, reward, attention))

    def sample(self, batch_size=32):
        return random.sample(self.memory, min(len(self.memory), batch_size))


# ═════════════════════════════════════════════════════════════════════════════
#  3. DATA INGESTION & ROBUST TELEMETRY
# ═════════════════════════════════════════════════════════════════════════════

class DataLoader:
    """Loads historical AutoCAD and SQLite telemetry with path resolution."""

    def __init__(self, csv_name="autocad_data.csv", db_name="collected_data.db"):
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()

        self.csv_path = os.path.join(base_dir, csv_name)
        self.db_path = os.path.join(base_dir, db_name)

        if not os.path.exists(self.csv_path): self.csv_path = csv_name
        if not os.path.exists(self.db_path): self.db_path = db_name

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
            print(f"📊 AutoCAD Ingested: {len(data)} frames.")
        return data

    def _load_db(self) -> List[Dict]:
        data = []
        if os.path.exists(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT energy, ore, semi_finished, finished_goods, propellant FROM planetary_economy LIMIT 5000")
                rows = cursor.fetchall()
                for r in rows:
                    data.append({"energy": r[0], "ore": r[1], "semi": r[2], "goods": r[3], "prop": r[4]})
                conn.close()
                print(f"💾 Economy DB Connected: {len(data)} rows.")
            except sqlite3.Error:
                pass
        return data


# ═════════════════════════════════════════════════════════════════════════════
#  4. THE STUDENT SNN
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

        # FIX: Brian 2 Equations must use newlines, not semicolons for parameter declarations
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
#  5. APEX TRAINING ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def run_apex_training():
    print("═" * 80)
    print("  🧠 PLANET FACTORY: PHASE 18 - APEX HYBRID SNN TRAINER")
    print("═" * 80)

    # 1. Load Edge Models & Distillation Teacher
    try:
        hf_model = torch.jit.load("student_distilled_heads_hf.torchscript.pt")
        basic_model = torch.jit.load("student_distilled_heads.torchscript.pt")
        print("✅ JIT Edge Models online.")
    except Exception:
        hf_model, basic_model = StudentDistilledHeadsHF(), StudentDistilledHeadsBasic()

    teacher = PyTorchTeacherHead().to(DEVICE)

    # Path resolution for weight injection
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    t_weights = os.path.join(base_dir, "robot_arm_snn_head.pt")
    if not os.path.exists(t_weights): t_weights = "robot_arm_snn_head.pt"

    try:
        teacher.load_state_dict(torch.load(t_weights, map_location=DEVICE), strict=False)
        teacher.eval()
        print(f"✅ Distillation Teacher online ({os.path.basename(t_weights)}).")
    except Exception as e:
        print(f"⚠️ Teacher offline: {e}")

    perception = PerceptionPipeline(FeatureNormalizer(NORM_JSON), HoloSynHeads(), hf_model, basic_model)
    student = StudentNeuromorphicHead(n_actions=3)
    data = DataLoader()
    memory = ExperienceMemory()

    # ── BUG FIX: Markovian Energy Tracker ──
    # Instead of pulling static energy from mock, we track a state that actions affect
    current_energy = 5000.0

    epochs = 250
    avg_reward = 0.0
    best_reward = -999.0
    best_weights = None

    print(f"\n🚀 Commencing Hybrid Knowledge Distillation ({epochs} Epochs)...")

    for epoch in range(1, epochs + 1):
        # ── SENSORY FETCH ──
        ac = data.autocad_data[epoch % len(data.autocad_data)] if data.autocad_data else {"geo_prec": 0.5,
                                                                                          "asm_eff": 0.5,
                                                                                          "prec_chg": 0.0}

        # If DB is offline, we use our local current_energy tracker
        econ_energy = data.economy_data[epoch % len(data.economy_data)][
            "energy"] if data.economy_data else current_energy

        # Inject periodic "Precision Shock" to test attention
        if epoch % 50 == 0: ac["prec_chg"] = 0.8

        feat_basic = torch.tensor([[ac["geo_prec"], ac["asm_eff"], econ_energy / 10000.0, 0.1, 0.1]],
                                  dtype=torch.float32).to(DEVICE)
        bundle = ModalityBundle(E_t_aug=torch.randn(1, 528), E_i=torch.zeros(1, 2048), E_a=torch.randn(1, 2048),
                                E_v=torch.randn(1, 2048), H=torch.zeros(1, 32, 3), feat_hf=torch.randn(1, 789),
                                feat_basic=feat_basic, mask=torch.ones(1, 4))

        fused_emb, _, _ = perception.perceive(bundle)

        # ── BUG FIX: REWARD SEQUENCING ──
        # 1. Predict Teacher action first
        with torch.no_grad():
            teacher_logits = teacher(fused_emb)
            teacher_action = torch.argmax(teacher_logits, dim=-1).item()

        # 2. Predict student action (without stepping Brian2 yet)
        # In a real SNN we run the step to get the action, then apply reward retrospectively.
        # But for training speed, we'll calculate the base reward components first.

        base_reward = (ac["asm_eff"] - 0.5) * 2.0
        if ac["geo_prec"] > 0.8: base_reward += 1.5
        if econ_energy < 1500: base_reward -= 2.0

        # ── DYNAMIC ATTENTION (Prediction Error Fix) ──
        reward_error = abs(base_reward - avg_reward)
        attention_signal = 1.0 + (reward_error * 2.0) + (abs(ac["prec_chg"]) * 10.0)
        avg_reward = 0.9 * avg_reward + 0.1 * base_reward

        # 3. Step the Student and get action
        student_action, weights = student.train_step(fused_emb, base_reward, attention_signal)

        # 4. Apply Distillation Bonus AFTER step (for next iteration) or integrated
        # To fix the bug, we ensure the teacher bonus is visible to the SNN.
        final_reward = base_reward
        if student_action == teacher_action:
            final_reward += 1.0  # Significant boost for matching instinct

        # ── ACTION PHYSICS (Update energy tracker) ──
        # Action 0: High Energy, Action 1: Med, Action 2: Rest/Recovery
        if student_action == 0:
            current_energy -= 100.0
        elif student_action == 1:
            current_energy -= 50.0
        else:
            current_energy += 200.0  # Recovery

        current_energy = max(0, min(10000.0, current_energy))

        memory.push(fused_emb, student_action, final_reward, attention_signal)

        # ── BIOLOGICAL DREAMING (Experience Replay) ──
        if epoch % 10 == 0 and len(memory.memory) > 32:
            batch = memory.sample(16)
            for m_state, m_act, m_rew, m_attn in batch:
                student.train_step(m_state, m_rew, m_attn)

        # Update best weights based on the final_reward (Corrected assignment)
        if final_reward > best_reward:
            best_reward = final_reward
            best_weights = weights.copy()

        if epoch % 25 == 0:
            print(
                f"  [Epoch {epoch:03d}] Act: {student_action} (Teach: {teacher_action}) | NRG: {econ_energy:4.0f} | Attn: {attention_signal:.2f} -> Rwd: {final_reward:+.1f}")

    print(f"\n🏁 Training Complete. Best Momentary Reward: {best_reward:.2f}")
    if best_weights is not None:
        torch.save({"synaptic_weights": best_weights}, "optimized_living_planet_weights.pt")
        print("💾 Optimized weights saved to optimized_living_planet_weights.pt")


if __name__ == "__main__":
    run_apex_training()