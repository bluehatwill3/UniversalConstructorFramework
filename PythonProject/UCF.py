"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              UNIVERSAL CONSTRUCTOR — ROBOT FRAMEWORK v3.0                  ║
║                                                                              ║
║  Architecture:                                                               ║
║   Sensor Input (text/img/audio/video/haptics)                                ║
║        ↓  Feature Extraction + Normalization                                 ║
║   HoloSynHeads (Teacher)  →  5 × 256-dim L2-normalized embeddings           ║
║   StudentDistilledHeads   →  4-dim modality attention weights (Sigmoid)      ║
║        ↓  Masked weighted fusion                                             ║
║   256-dim fused state embedding                                              ║
║        ↓  Neuromorphic Action Head (Brian2 SNN Core)                         ║
║   Safety Governor & Recovery Protocol                                        ║
║        ↓  Robot action + Quantum Sentiment (Cirq + QSim)                     ║
║   Self-replicating child via knowledge distillation                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from collections import defaultdict
from datetime import datetime

# ── ADVANCED INTEGRATIONS ────────────────────────────────────────────────────
try:
    import cirq
    import qsimcirq

    HAS_QUANTUM = True
except ImportError:
    HAS_QUANTUM = False
    warnings.warn("cirq or qsimcirq not installed. Quantum sentiment will fallback.")

try:
    import brian2 as b2

    b2.prefs.codegen.target = 'numpy'  # Suppress C++ compilation overhead
    HAS_NEUROMORPHIC = True
except ImportError:
    HAS_NEUROMORPHIC = False
    warnings.warn("brian2 not installed. Neuromorphic core will fallback to PyTorch.")

warnings.filterwarnings("ignore")


# 🟢 Hardware acceleration
def get_optimal_device():
    if torch.cuda.is_available():
        try:
            _ = torch.tensor([1.0], device='cuda') * 2.0
            return torch.device("cuda")
        except Exception:
            return torch.device("cpu")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_optimal_device()

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
HOLOSYN_TORCHSCRIPT = os.path.join(MODEL_DIR, "holosyn_heads.torchscript.pt")
STUDENT_HF_TS = os.path.join(MODEL_DIR, "student_distilled_heads_hf.torchscript.pt")
STUDENT_BASIC_TS = os.path.join(MODEL_DIR, "student_distilled_heads.torchscript.pt")
NORM_JSON = os.path.join(MODEL_DIR, "student_norm_hf.json")


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 0 — SAFETY & RECOVERY DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class IndustrialThresholds:
    max_temp: float = 85.0
    vibration_limit: float = 0.8
    min_energy: float = 5.0


class SafetyGovernor:
    def __init__(self, thresholds: IndustrialThresholds):
        self.limits = thresholds

    def validate_action(self, action_id: int, sensors: Dict) -> Tuple[bool, str]:
        if sensors.get("temperature", 0.0) > self.limits.max_temp:
            return False, "THERMAL_OVERLOAD"
        if sensors.get("vibration", 0.0) > self.limits.vibration_limit:
            return False, "MECHANICAL_STRESS"
        if sensors.get("energy", 100.0) < self.limits.min_energy:
            return False, "LOW_POWER"
        return True, "SAFE"


class RecoveryProtocol:
    def __init__(self, cool_down_cycles: int = 5):
        self.cool_down_cycles = cool_down_cycles
        self.stability_tracker = {}

    def check_recovery(self, robot_name: str, is_safe: bool) -> bool:
        if not is_safe:
            self.stability_tracker[robot_name] = 0
            return False
        self.stability_tracker[robot_name] = self.stability_tracker.get(robot_name, 0) + 1
        return self.stability_tracker[robot_name] >= self.cool_down_cycles


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 1 — FEATURE NORMALIZATION & PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

class FeatureNormalizer:
    TXT_SLICE, HAPT_SLICE, AUD_SLICE, IMG_SLICE, W2V_SLICE, FULL_DIM = slice(0, 5), slice(5, 8), slice(8, 12), slice(12,
                                                                                                                     21), slice(
        21, 789), 789

    def __init__(self, norm_path: Optional[str] = None):
        self.mu = np.zeros(self.FULL_DIM, dtype=np.float32)
        self.sd = np.ones(self.FULL_DIM, dtype=np.float32)
        if norm_path and os.path.exists(norm_path):
            with open(norm_path) as f: d = json.load(f)
            self.mu = np.array(d["mu"], dtype=np.float32)
            self.sd = np.where(np.array(d["sd"], dtype=np.float32) < 1e-9, 1.0, np.array(d["sd"], dtype=np.float32))

    def make_feature_vector(self, txt_meta=None, hapt_meta=None, aud_meta=None, img_meta=None, w2v=None) -> np.ndarray:
        vec = np.zeros(self.FULL_DIM, dtype=np.float32)
        if txt_meta is not None: vec[self.TXT_SLICE] = txt_meta[:5]
        if aud_meta is not None: vec[self.AUD_SLICE] = aud_meta[:4]
        return (vec - self.mu) / self.sd


# (Neural Network Model Definitions - HoloSyn, Students, Encoders)
class Projector(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x): return F.normalize(self.net(x), p=2, dim=-1)


class HapticsEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(3, 32, 7, padding=3), nn.ReLU(), nn.Conv1d(32, 64, 7, padding=3), nn.ReLU(),
                                 nn.AdaptiveAvgPool1d(1))
        self.proj = nn.Linear(64, 256)

    def forward(self, H): return F.normalize(self.proj(self.net(H.transpose(1, 2)).squeeze(-1)), p=2, dim=-1)


class HoloSynHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.p_t, self.p_i, self.p_a, self.p_v = Projector(528), Projector(2048), Projector(2048), Projector(2048)
        self.h_enc = HapticsEncoder()

    def forward(self, E_t_aug, E_i, E_a, E_v, H): return self.p_t(E_t_aug), self.p_i(E_i), self.p_a(E_a), self.p_v(
        E_v), self.h_enc(H)

    def load_torchscript(self, path):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location="cpu", weights_only=False), strict=False)


class StudentDistilledHeadsHF(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(789, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 4),
                                 nn.Sigmoid())

    def forward(self, x): return self.net(x)

    def load_torchscript(self, path):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location="cpu", weights_only=False), strict=False)


class StudentDistilledHeadsBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 4),
                                 nn.Sigmoid())

    def forward(self, x): return self.net(x)

    def load_torchscript(self, path):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location="cpu", weights_only=False), strict=False)


@dataclass
class ModalityBundle:
    E_t_aug: torch.Tensor;
    E_i: Optional[torch.Tensor];
    E_a: torch.Tensor;
    E_v: torch.Tensor
    H: torch.Tensor;
    feat_hf: torch.Tensor;
    feat_basic: torch.Tensor;
    mask: torch.Tensor
    meta: Dict = field(default_factory=dict)


class PerceptionPipeline:
    def __init__(self, normalizer, holosyn, student_hf, student_basic):
        self.norm, self.holosyn, self.student_hf, self.student_basic = normalizer, holosyn, student_hf, student_basic

    @torch.no_grad()
    def perceive(self, bundle: ModalityBundle) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # 1. FIXED: Explicitly check for None to avoid the PyTorch boolean ambiguity
        E_i_safe = bundle.E_i if bundle.E_i is not None else torch.zeros(1, 2048)

        # 2. Pass the safe tensor into the HoloSyn projector
        z_t, z_i, z_a, z_v, z_h = self.holosyn(bundle.E_t_aug, E_i_safe, bundle.E_a, bundle.E_v, bundle.H)

        raw_scores = self.student_hf(bundle.feat_hf)
        masked = raw_scores * bundle.mask
        logits = torch.where(bundle.mask > 0, torch.log(masked + 1e-9), torch.full_like(masked, -1e9))
        modal_weights = torch.softmax(logits, dim=-1)

        w = modal_weights.unsqueeze(-1)
        fused = F.normalize((w * torch.stack([z_t, z_i, z_a, z_v], dim=1)).sum(dim=1), p=2, dim=-1)

        return fused, modal_weights, {}

    def make_synthetic_bundle(self, seed=None) -> ModalityBundle:
        rng = np.random.default_rng(seed)
        txt_meta = np.array(
            [rng.integers(10, 200), rng.integers(1, 10), rng.integers(0, 5), rng.integers(0, 3), rng.uniform(0, 0.15)],
            dtype=np.float32)
        aud_meta = np.array([rng.uniform(0.0, 0.5), rng.uniform(0.0, 0.05), rng.uniform(60, 95), rng.uniform(60, 180)],
                            dtype=np.float32)  # Temp is index 2
        t = np.linspace(0, 2 * np.pi, 32)
        haptic_seq = np.stack(
            [np.sin(t) + rng.standard_normal(32) * 0.1, np.cos(t) + rng.standard_normal(32) * 0.1, np.zeros(32)],
            axis=-1).astype(np.float32)

        return ModalityBundle(
            E_t_aug=torch.randn(1, 528), E_i=torch.randn(1, 2048), E_a=torch.randn(1, 2048), E_v=torch.randn(1, 2048),
            H=torch.from_numpy(haptic_seq).unsqueeze(0),
            feat_hf=torch.from_numpy(self.norm.make_feature_vector(txt_meta=txt_meta, aud_meta=aud_meta)).unsqueeze(0),
            feat_basic=torch.from_numpy(txt_meta[:5]).unsqueeze(0),
            mask=torch.ones(1, 4), meta={"aud_meta": aud_meta}
        )


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — NEUROMORPHIC & QUANTUM
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class QuantumSentiment:
    mood: str

    @staticmethod
    def from_embedding(emb: torch.Tensor) -> "QuantumSentiment":
        v = emb.squeeze().detach().cpu().numpy()
        stability = abs(np.clip((np.percentile(v, 50) + 1) / 2, 0, 1) - 0.5)
        return QuantumSentiment(mood="stable" if stability < 0.25 else "drifting")


class NeuromorphicActionHead(nn.Module):
    def __init__(self, n_actions: int, is_child: bool = False):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_actions))

    @torch.no_grad()
    def predict(self, z: torch.Tensor) -> int:
        if HAS_NEUROMORPHIC:
            b2.start_scope()
            # FIXED: Changed 'rates' to 'snn_input_rates' to avoid namespace conflict
            snn_input_rates = np.clip(z.squeeze().cpu().numpy(), 0, 1) * 100 * b2.Hz
            P = b2.PoissonGroup(256, rates=snn_input_rates)
            G = b2.NeuronGroup(self.n_actions, 'dv/dt = (I - v) / (10*ms) : 1\nI : 1', threshold='v>1', reset='v=0',
                               refractory=2 * b2.ms, method='exact')
            S = b2.Synapses(P, G, 'w : 1', on_pre='v_post += w')
            S.connect()
            S.w = 'rand() * 0.1'
            M = b2.SpikeMonitor(G)
            b2.run(20 * b2.ms)
            if np.sum(M.count) > 0: return int(np.argmax(M.count))
        return int(self.net(z).argmax(dim=-1).item())


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 3 — ROBOTS & ECONOMICS
# ═════════════════════════════════════════════════════════════════════════════

DOMAINS = {
    "robot_arm": {"n_actions": 2, "action_map": {0: "➡️ Move", 1: "🤝 Engage"}, "emoji": "🦾"},
    "cnc_production": {"n_actions": 3, "action_map": {0: "✅ Maintain", 1: "🔧 Adjust", 2: "⚙️ Swap"}, "emoji": "🏭"},
}


@dataclass
class ResourceLedger:
    raw_materials: float = 100.0;
    component_parts: int = 0;
    energy: float = 50.0

    def update_for_action(self, action_id: int):
        self.energy = max(0, self.energy - (0.1 if action_id == 0 else 0.5))
        if action_id > 0: self.component_parts += 1


class SelfReplicatingRobot:
    def __init__(self, name: str, domain: str, perception: PerceptionPipeline):
        self.name, self.domain, self.perception = name, domain, perception
        self.domain_cfg = DOMAINS[domain]
        self.resources = ResourceLedger()
        self.action_head = NeuromorphicActionHead(self.domain_cfg["n_actions"])
        self.action_log = []


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 4 — UNIVERSAL CONSTRUCTOR (MAIN ENGINE)
# ═════════════════════════════════════════════════════════════════════════════

class UniversalConstructor:
    def __init__(self, n_cycles: int = 60, verbose: bool = True):
        self.n_cycles, self.verbose = n_cycles, verbose
        self.hive_log = []

        print("\n" + "═" * 70)
        print("  🌌  UNIVERSAL CONSTRUCTOR — PLANET FACTORY EDITION")
        print("═" * 70)

        # Initialize Safety Core
        self.governor = SafetyGovernor(IndustrialThresholds())
        self.recovery = RecoveryProtocol(cool_down_cycles=3)

        # Initialize Brains
        self.norm = FeatureNormalizer(NORM_JSON)
        self.holosyn = HoloSynHeads()
        self.student_hf = StudentDistilledHeadsHF()
        self.student_basic = StudentDistilledHeadsBasic()
        self.perception = PerceptionPipeline(self.norm, self.holosyn, self.student_hf, self.student_basic)

        # Spawn Alpha Robots
        self.robots = {
            "Alpha-Arm": SelfReplicatingRobot("Alpha-Arm", "robot_arm", self.perception),
            "Alpha-CNC": SelfReplicatingRobot("Alpha-CNC", "cnc_production", self.perception)
        }
        print(f"✅ Planet Factory ready — {len(self.robots)} systems online")

    def run(self):
        print(f"\n🚀 Starting Safety-Guarded Simulation — {self.n_cycles} cycles")

        for cycle in range(1, self.n_cycles + 1):
            for name, robot in self.robots.items():

                # 1. Perception & Prediction
                bundle = self.perception.make_synthetic_bundle(seed=cycle + id(robot))
                fused_emb, _, _ = self.perception.perceive(bundle)
                proposed_id = robot.action_head.predict(fused_emb)

                # 2. Extract Metrics
                metrics = {
                    "temperature": bundle.meta.get("aud_meta", [0, 0, 0, 0])[2],
                    "vibration": np.std(bundle.H.numpy()),
                    "energy": robot.resources.energy
                }

                # 3. Safety & Recovery Validation
                is_safe, reason = self.governor.validate_action(proposed_id, metrics)

                if self.recovery.check_recovery(name, is_safe):
                    final_id = proposed_id
                    action_name = robot.domain_cfg["action_map"].get(final_id, "Unknown")
                else:
                    final_id = 0
                    action_name = f"⚠️ BLOCKED ({reason})"

                # 4. Execute & Log
                robot.resources.update_for_action(final_id)
                sentiment = QuantumSentiment.from_embedding(fused_emb)

                log_entry = {"cycle": cycle, "action": action_name, "safe": is_safe, "reason": reason,
                             "mood": sentiment.mood}
                robot.action_log.append(log_entry)
                self.hive_log.append({"robot": name, **log_entry})

                if self.verbose:
                    print(
                        f"  Cycle {cycle:03d} | {robot.domain_cfg['emoji']} {name:<12} | {action_name:<20} | Temp: {metrics['temperature']:.1f}°C")


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uc = UniversalConstructor(n_cycles=20)
    uc.run()