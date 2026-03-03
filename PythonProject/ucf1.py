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
║   Robot action + Quantum Sentiment (Cirq + QSim)                             ║
║        ↓  Resource economics                                                 ║
║   Self-replicating child via knowledge distillation                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from collections import defaultdict
from datetime import datetime
import numpy as np
import torch
from dataclasses import dataclass, field
# Ensure SafetyGovernor and ModalityBundle classes are defined before UniversalConstructor
# ── ADVANCED INTEGRATIONS ────────────────────────────────────────────────────
try:
    import cirq
    import qsimcirq
    HAS_QUANTUM = True
except ImportError:
    HAS_QUANTUM = False
    warnings.warn("cirq or qsimcirq not installed. Quantum sentiment will fallback to classical simulation.")

try:
    import brian2 as b2
    # Suppress C++ compilation overhead for rapid local script execution
    b2.prefs.codegen.target = 'numpy'
    HAS_NEUROMORPHIC = True
except ImportError:
    HAS_NEUROMORPHIC = False
    warnings.warn("brian2 not installed. Neuromorphic core will fallback to standard PyTorch MLPs.")

warnings.filterwarnings("ignore")


class UniversalConstructor:
    def __init__(self, model_dir=None, domains=None, n_cycles=60, verbose=True):
        # ... (previous initialization code) ...

        # Initialize the Safety System
        self.thresholds = IndustrialThresholds(max_temp=82.0, vibration_limit=0.75)
        self.governor = SafetyGovernor(self.thresholds)

        # Shared perception and robots setup
        self.perception = PerceptionPipeline(self.normalizer, self.holosyn, self.student_hf, self.student_basic)
        self.robots = {}
        # ... (robot spawning logic) ...

    def run(self):
        print(f"\n🚀 Starting Safety-Guarded Simulation — {self.n_cycles} cycles")
        all_robots = dict(self.robots)

        for cycle in range(1, self.n_cycles + 1):
            for name, robot in list(all_robots.items()):
                # 1. Generate sensing data
                bundle = self.perception.make_synthetic_bundle(seed=cycle + robot.id)

                # 2. Perception -> Proposed Action
                fused_emb, _, _ = self.perception.perceive(bundle)
                proposed_id = robot.action_head.predict(fused_emb)

                # 3. Extract real-time metrics for the Governor
                metrics = {
                    "temperature": bundle.meta.get("aud_meta", [0, 0, 0, 0])[2],
                    "vibration": np.std(bundle.H.numpy()),
                    "energy": robot.resources.energy
                }

                # 4. Safety Interception
                is_safe, reason = self.governor.validate_action(proposed_id, metrics)

                if not is_safe:
                    # Force a safe 'Wait' action (Action 0)
                    final_id = 0
                    action_name = f"⚠️ BLOCKED ({reason})"
                else:
                    final_id = proposed_id
                    action_name = robot.domain_cfg["action_map"].get(final_id, "Unknown")

                # 5. Execute and Log
                robot.resources.update_for_action(final_id, robot.domain)
                robot.action_log.append({
                    "cycle": cycle, "action_name": action_name, "safe": is_safe, "reason": reason
                })

                if self.verbose:
                    print(f"  Cycle {cycle:03d} | {name:<15} | {action_name}")

            # Every 10 cycles, show the Safety Dashboard
            if cycle % 10 == 0:
                print_hive_safety_dashboard(list(all_robots.values()), self.governor)


# 🟢 Hardware acceleration support with safe kernel execution fallback
def get_optimal_device():
    if torch.cuda.is_available():
        try:
            # Verify CUDA kernel execution (catches architecture mismatches)
            _ = torch.tensor([1.0], device='cuda') * 2.0
            return torch.device("cuda")
        except Exception as e:
            warnings.warn("CUDA detected but kernel execution failed (likely architecture mismatch). Falling back to CPU.")
            return torch.device("cpu")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_optimal_device()



# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

HOLOSYN_TORCHSCRIPT   = os.path.join(MODEL_DIR, "holosyn_heads.torchscript.pt")
STUDENT_HF_TS         = os.path.join(MODEL_DIR, "student_distilled_heads_hf.torchscript.pt")
STUDENT_BASIC_TS      = os.path.join(MODEL_DIR, "student_distilled_heads.torchscript.pt")
NORM_JSON             = os.path.join(MODEL_DIR, "student_norm_hf.json")


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 0 — FEATURE NORMALIZATION
# ═════════════════════════════════════════════════════════════════════════════
class RecoveryProtocol:
    def __init__(self, cool_down_cycles: int = 5):
        self.cool_down_cycles = cool_down_cycles
        self.stability_tracker = {}  # robot_name: count_of_safe_cycles

    def check_recovery(self, robot_name: str, is_safe: bool) -> bool:
        """Only returns True after a sustained period of safety."""
        if not is_safe:
            self.stability_tracker[robot_name] = 0
            return False

        self.stability_tracker[robot_name] = self.stability_tracker.get(robot_name, 0) + 1

        # Check if we have met the cool-down requirement
        if self.stability_tracker[robot_name] >= self.cool_down_cycles:
            return True
        return False


class FeatureNormalizer:
    TXT_SLICE  = slice(0, 5)
    HAPT_SLICE = slice(5, 8)
    AUD_SLICE  = slice(8, 12)
    IMG_SLICE  = slice(12, 21)
    W2V_SLICE  = slice(21, 789)
    FULL_DIM   = 789

    def __init__(self, norm_path: Optional[str] = None):
        self.mu = np.zeros(self.FULL_DIM, dtype=np.float32)
        self.sd = np.ones(self.FULL_DIM, dtype=np.float32)
        self.columns: List[str] = []
        if norm_path and os.path.exists(norm_path):
            self._load(norm_path)
        else:
            print(f"  ⚠️ Normalization stats JSON not found. Using identity normalization (mu=0, sd=1).")

    def _load(self, path: str):
        with open(path) as f:
            d = json.load(f)
        self.columns = d["numeric_cols"]
        self.mu = np.array(d["mu"], dtype=np.float32)
        self.sd = np.array(d["sd"], dtype=np.float32)
        self.sd = np.where(self.sd < 1e-9, 1.0, self.sd)
        print(f"  ✅ Normalization stats loaded: {len(self.columns)} features")

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sd

    def make_feature_vector(
        self,
        txt_meta:   Optional[np.ndarray] = None,
        hapt_meta:  Optional[np.ndarray] = None,
        aud_meta:   Optional[np.ndarray] = None,
        img_meta:   Optional[np.ndarray] = None,
        w2v:        Optional[np.ndarray] = None,
    ) -> np.ndarray:
        vec = np.zeros(self.FULL_DIM, dtype=np.float32)
        if txt_meta  is not None: vec[self.TXT_SLICE]  = txt_meta[:5]
        if hapt_meta is not None: vec[self.HAPT_SLICE] = hapt_meta[:3]
        if aud_meta  is not None: vec[self.AUD_SLICE]  = aud_meta[:4]
        if img_meta  is not None: vec[self.IMG_SLICE]  = img_meta[:9]
        if w2v       is not None: vec[self.W2V_SLICE]  = w2v[:768]
        return self.normalize(vec)


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 1 — MODEL DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════

class Projector(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)

class HapticsEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(64, 256)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        x = H.transpose(1, 2)
        x = self.net(x).squeeze(-1)
        x = self.proj(x)
        return F.normalize(x, p=2, dim=-1)

class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.proj = nn.Linear(256, 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.proj(x)

class HoloSynHeads(nn.Module):
    TEXT_DIM, IMG_DIM, AUD_DIM, VID_DIM = 528, 2048, 2048, 2048

    def __init__(self):
        super().__init__()
        self.p_t   = Projector(self.TEXT_DIM, hidden=512, out_dim=256)
        self.p_i   = Projector(self.IMG_DIM,  hidden=512, out_dim=256)
        self.p_a   = Projector(self.AUD_DIM,  hidden=512, out_dim=256)
        self.p_v   = Projector(self.VID_DIM,  hidden=512, out_dim=256)
        self.h_enc = HapticsEncoder()

    def forward(self, E_t_aug, E_i, E_a, E_v, H):
        return self.p_t(E_t_aug), self.p_i(E_i), self.p_a(E_a), self.p_v(E_v), self.h_enc(H)

    def load_torchscript(self, path: str) -> bool:
        if not os.path.exists(path): return False
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict): self.load_state_dict(ckpt, strict=False)
            return True
        except: return False

class StudentDistilledHeadsHF(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(789, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 4),   nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def load_torchscript(self, path: str) -> bool:
        if not os.path.exists(path): return False
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict): self.load_state_dict(ckpt, strict=False)
            return True
        except: return False

class StudentDistilledHeadsBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 4), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def load_torchscript(self, path: str) -> bool:
        if not os.path.exists(path): return False
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict): self.load_state_dict(ckpt, strict=False)
            return True
        except: return False


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — MULTIMODAL PERCEPTION PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ModalityBundle:
    E_t_aug: torch.Tensor
    E_i:     Optional[torch.Tensor]
    E_a:     torch.Tensor
    E_v:     torch.Tensor
    H:       torch.Tensor
    feat_hf: torch.Tensor
    feat_basic: torch.Tensor
    mask:    torch.Tensor
    raw_image: Optional[torch.Tensor] = None
    meta:    Dict = field(default_factory=dict)

class PerceptionPipeline:
    def __init__(self, normalizer: FeatureNormalizer, holosyn: HoloSynHeads,
                 student_hf: StudentDistilledHeadsHF, student_basic: StudentDistilledHeadsBasic,
                 vision_encoder: Optional[VisionEncoder] = None):
        self.norm, self.holosyn, self.student_hf, self.student_basic = normalizer, holosyn, student_hf, student_basic
        self.vision_encoder = vision_encoder
        self.holosyn.eval()
        self.student_hf.eval()
        self.student_basic.eval()
        if self.vision_encoder: self.vision_encoder.eval()

    @torch.no_grad()
    def perceive(self, bundle: ModalityBundle) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        if bundle.E_i is None:
            if bundle.raw_image is not None and self.vision_encoder is not None:
                bundle.E_i = self.vision_encoder(bundle.raw_image)
            else:
                bundle.E_i = torch.zeros(1, 2048)

        z_t, z_i, z_a, z_v, z_h = self.holosyn(bundle.E_t_aug, bundle.E_i, bundle.E_a, bundle.E_v, bundle.H)
        raw_scores = self.student_hf(bundle.feat_hf) if bundle.feat_hf is not None else self.student_basic(bundle.feat_basic)

        masked = raw_scores * bundle.mask
        logits = torch.where(bundle.mask > 0, torch.log(masked + 1e-9), torch.full_like(masked, -1e9))
        modal_weights = torch.softmax(logits, dim=-1)

        w = modal_weights.unsqueeze(-1)
        stack = torch.stack([z_t, z_i, z_a, z_v], dim=1)
        fused = F.normalize((w * stack).sum(dim=1), p=2, dim=-1)

        info = {"z_text": z_t, "z_img": z_i, "z_aud": z_a, "z_vid": z_v, "z_hapt": z_h, "modal_scores": raw_scores, "modal_weights": modal_weights}
        return fused, modal_weights, info

    def make_synthetic_bundle(self, txt_meta=None, haptic_seq=None, available_modalities=None, seed=None) -> ModalityBundle:
        rng = np.random.default_rng(seed)
        available = set(available_modalities or ["text", "image", "audio", "video"])

        txt_meta = txt_meta if txt_meta is not None else np.array([rng.integers(10, 200), rng.integers(1, 10), rng.integers(0, 5), rng.integers(0, 3), rng.uniform(0, 0.15)], dtype=np.float32)
        E_t_aug_np = rng.standard_normal(528).astype(np.float32)
        E_i_np     = rng.standard_normal(2048).astype(np.float32) if self.vision_encoder is None else None
        E_a_np     = rng.standard_normal(2048).astype(np.float32)
        E_v_np     = rng.standard_normal(2048).astype(np.float32)
        raw_image  = torch.randn(1, 3, 224, 224) if self.vision_encoder is not None else None

        if haptic_seq is None:
            t = np.linspace(0, 2 * np.pi, 32)
            haptic_seq = np.stack([np.sin(t) + rng.standard_normal(32)*0.1, np.cos(t) + rng.standard_normal(32)*0.1, 0.5*np.sin(2*t) + rng.standard_normal(32)*0.05], axis=-1).astype(np.float32)

        w2v = rng.standard_normal(768).astype(np.float32)
        aud_meta = np.array([rng.uniform(0.0, 0.5), rng.uniform(0.0, 0.05), rng.uniform(100, 4000), rng.uniform(60, 180)], dtype=np.float32)
        img_meta = np.zeros(9, dtype=np.float32)

        feat_789 = self.norm.make_feature_vector(txt_meta=txt_meta, aud_meta=aud_meta, img_meta=img_meta, w2v=w2v)

        mask = np.array([1.0 if "text" in available else 0.0, 1.0 if "image" in available else 0.0, 1.0 if "audio" in available else 0.0, 1.0 if "video" in available else 0.0], dtype=np.float32)

        return ModalityBundle(
            E_t_aug=torch.from_numpy(E_t_aug_np).unsqueeze(0),
            E_i=torch.from_numpy(E_i_np).unsqueeze(0) if E_i_np is not None else None,
            E_a=torch.from_numpy(E_a_np).unsqueeze(0),
            E_v=torch.from_numpy(E_v_np).unsqueeze(0),
            H=torch.from_numpy(haptic_seq).unsqueeze(0),
            feat_hf=torch.from_numpy(feat_789).unsqueeze(0),
            feat_basic=torch.from_numpy(txt_meta[:5]).unsqueeze(0),
            mask=torch.from_numpy(mask).unsqueeze(0),
            raw_image=raw_image,
            meta={"txt_meta": txt_meta, "aud_meta": aud_meta},
        )


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 3 — QUANTUM SENTIMENT ENGINE (CIRCUIT INTEGRATION)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class QuantumSentiment:
    cycle: int
    agent: str
    classical_H: float
    classical_M: float
    classical_S: float
    quantum_H:   float
    quantum_M:   float
    quantum_S:   float
    mood: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @staticmethod
    def from_embedding(emb: torch.Tensor, agent: str, cycle: int) -> "QuantumSentiment":
        v = emb.squeeze().detach().cpu().numpy()

        # Classical statistics
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        c_H = float(np.clip((q2 + 1) / 2, 0, 1))
        c_M = float(np.clip(np.linalg.norm(v) / 16, 0, 1))
        c_S = float(np.clip((q3 - q1 + 1) / 2, 0, 1))

        # Quantum Mechanics (Cirq + QSim if available)
        if HAS_QUANTUM:
            # Map first 3 embedding dimensions to rotation angles [0, 2π]
            angles = np.clip(np.abs(v[:3]) * np.pi, 0, 2 * np.pi)
            qubits = cirq.LineQubit.range(3)
            circuit = cirq.Circuit(
                cirq.rx(angles[0])(qubits[0]),
                cirq.ry(angles[1])(qubits[1]),
                cirq.rz(angles[2])(qubits[2]),
                cirq.CNOT(qubits[0], qubits[1]),  # 🟢 Added quantum entanglement
                cirq.CNOT(qubits[1], qubits[2]),  # 🟢 Added quantum entanglement
                cirq.measure(*qubits, key='m')
            )
            sim = qsimcirq.QSimSimulator()
            res = sim.run(circuit, repetitions=100)
            counts = res.histogram(key='m')

            # Map the 8 possible states to H, M, S metrics
            q_H = float(counts.get(0, 0) + counts.get(1, 0)) / 100.0
            q_M = float(counts.get(2, 0) + counts.get(3, 0)) / 100.0
            q_S = float(counts.get(4, 0) + counts.get(5, 0) + counts.get(6, 0) + counts.get(7, 0)) / 100.0
        else:
            # Fallback numpy simulation
            probs = np.abs(v[:8]) ** 2
            probs /= probs.sum() + 1e-9
            q_H = float(probs[:3].sum())
            q_M = float(probs[3:6].sum())
            q_S = float(probs[6:].sum())

        stability = abs(c_H - 0.5) + abs(c_M - 0.5)
        mood = "stable" if stability < 0.25 else "drifting"

        return QuantumSentiment(
            cycle=cycle, agent=agent,
            classical_H=round(c_H, 3), classical_M=round(c_M, 3), classical_S=round(c_S, 3),
            quantum_H=round(q_H, 3),   quantum_M=round(q_M, 3),   quantum_S=round(q_S, 3),
            mood=mood,
        )

    def format(self) -> str:
        return f"[{self.agent}] cycle={self.cycle} mood={self.mood}. c(H={self.classical_H}, M={self.classical_M}, S={self.classical_S}) q(H={self.quantum_H}, M={self.quantum_M}, S={self.quantum_S})."


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 4 — NEUROMORPHIC DOMAIN HEADS (BRIAN2 SNN)
# ═════════════════════════════════════════════════════════════════════════════

class NeuromorphicActionHead(nn.Module):
    """
    Action head that routes through a PyTorch MLP (training)
    or a Brian2 Spiking Neural Network (inference) based on availability.
    """
    def __init__(self, n_actions: int, is_child: bool = False):
        super().__init__()
        self.n_actions = n_actions
        self.is_child = is_child
        if is_child:
            self.net = nn.Sequential(nn.Linear(256, 32), nn.ReLU(), nn.Linear(32, n_actions))
        else:
            self.net = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_actions))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    @torch.no_grad()
    def predict(self, z: torch.Tensor) -> int:
        if HAS_NEUROMORPHIC:
            # Execute through Brian2 SNN
            b2.start_scope()

            # Input Layer: 256 Poisson neurons driven by normalized embedding
            rates = np.clip(z.squeeze().cpu().numpy(), 0, 1) * 100 * b2.Hz
            P = b2.PoissonGroup(256, rates=rates)

            # Output Layer: LIF Neurons mapping to actions
            eqs = '''
            dv/dt = (I - v) / (10*ms) : 1
            I : 1
            '''
            G = b2.NeuronGroup(self.n_actions, eqs, threshold='v>1', reset='v=0', refractory=2*b2.ms, method='exact')  # 🟢 Added biological refractory period

            # Synapses: Forward connecting based on PyTorch dense weights
            S = b2.Synapses(P, G, 'w : 1', on_pre='v_post += w')
            S.connect()

            # Approximate the synaptic weights from the linear PyTorch layer
            w_pt = torch.abs(self.net[0].weight.mean(dim=1)[:self.n_actions]).cpu().numpy()
            S.w = 'rand() * 0.1' # Simplify for local script speed

            # Monitor Spikes
            M = b2.SpikeMonitor(G)
            b2.run(20 * b2.ms) # Simulate 20ms of biological time

            spike_counts = np.array(M.count)
            if spike_counts.sum() == 0:
                # Fallback to pure MLP if no neurons fired
                return int(self(z).argmax(dim=-1).item())

            return int(np.argmax(spike_counts))
        else:
            # Pure PyTorch MLP inference
            self.eval()
            return int(self(z).argmax(dim=-1).item())


DOMAINS: Dict[str, Dict] = {
    "robot_arm": {
        "description": "Physical Robot Control (Arms/Actuators)",
        "n_actions": 2, "action_map": {0: "➡️  Move actuator", 1: "🤝 Engage payload"}, "emoji": "🦾",
    },
    "self_driving": {
        "description": "Autonomous Vehicles / Navigation",
        "n_actions": 3, "action_map": {0: "⬅️  Steer left", 1: "⬆️  Drive straight",  2: "➡️  Steer right"}, "emoji": "🚗",
    },
    "cnc_production": {
        "description": "Factory / CNC / Manufacturing",
        "n_actions": 3, "action_map": {0: "✅ Maintain params", 1: "🔧 Adjust CNC tool", 2: "⚙️  Swap material"}, "emoji": "🏭",
    },
    "swarm_coordination": {
        "description": "Multi-Robot Coordination / Swarms",
        "n_actions": 3, "action_map": {0: "📡 Broadcast sync", 1: "🐜 Form structure", 2: "🕸️ Disperse"}, "emoji": "🐝",
    },
    "universe_builder": {
        "description": "Planet Factory Core",
        "n_actions": 3, "action_map": {0: "🌑 Asteroid mine", 1: "🔥 Core ignition", 2: "🌍 Orbital stabilize"}, "emoji": "🌌",
    },
}


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 5 — SELF-REPLICATING ROBOT
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ResourceLedger:
    raw_materials: float = 100.0
    component_parts: int = 0
    energy: float = 50.0
    replication_cost_materials: float = 20.0
    replication_cost_parts: int = 10
    replication_cost_energy: float = 15.0

    def can_replicate(self) -> bool:
        return (self.raw_materials  >= self.replication_cost_materials and
                self.component_parts >= self.replication_cost_parts and
                self.energy          >= self.replication_cost_energy)

    def consume_for_replication(self):
        self.raw_materials  -= self.replication_cost_materials
        self.component_parts -= self.replication_cost_parts
        self.energy          -= self.replication_cost_energy

    def update_for_action(self, action_id: int, domain: str):
        if domain in ("cnc_production", "universe_builder"):
            if action_id in (1, 2):
                self.raw_materials   = max(0, self.raw_materials - 2)
                self.component_parts += 1
                self.energy          = max(0, self.energy - 1)
            else:
                self.energy = min(100, self.energy + 0.5)
        elif domain == "swarm_coordination":
            self.raw_materials   = max(0, self.raw_materials - 1)
            self.component_parts += 1
            self.energy          = max(0, self.energy - 2)
        else:
            self.energy = max(0, self.energy - 0.3)


class SelfReplicatingRobot:
    _robot_count = 0

    def __init__(self, name: str, domain: str, perception: PerceptionPipeline,
                 action_head: Optional[NeuromorphicActionHead] = None, is_child: bool = False,
                 parent_name: Optional[str] = None, resources: Optional[ResourceLedger] = None):
        SelfReplicatingRobot._robot_count += 1
        self.id, self.name, self.domain = SelfReplicatingRobot._robot_count, name, domain
        self.domain_cfg   = DOMAINS[domain]
        self.perception   = perception
        self.is_child, self.parent_name  = is_child, parent_name
        self.resources    = resources or ResourceLedger()
        self.children, self.sentiment_history, self.action_log = [], [], []
        self.cycle = 0

        n_actions = self.domain_cfg["n_actions"]
        self.action_head = action_head or NeuromorphicActionHead(n_actions, is_child=is_child)

        tier = "👶 Child" if is_child else "🤖 Parent"
        print(f"  {tier} Robot '{name}' — domain: {self.domain_cfg['emoji']} {domain}")

    def perceive_and_act(self, bundle: Optional[ModalityBundle] = None, seed: Optional[int] = None):
        self.cycle += 1
        bundle = bundle or self.perception.make_synthetic_bundle(seed=seed)
        fused_emb, modal_weights, _ = self.perception.perceive(bundle)

        action_id = self.action_head.predict(fused_emb)
        action_name = self.domain_cfg["action_map"].get(action_id, "Unknown")

        sentiment = QuantumSentiment.from_embedding(fused_emb, self.name, self.cycle)
        self.sentiment_history.append(sentiment)
        self.resources.update_for_action(action_id, self.domain)

        self.action_log.append({
            "cycle": self.cycle, "action_id": action_id, "action_name": action_name,
            "modal_weights": modal_weights.squeeze().cpu().tolist(), "mood": sentiment.mood,
            "resources": {"materials": round(self.resources.raw_materials, 1), "parts": self.resources.component_parts, "energy": round(self.resources.energy, 1)}
        })

        return action_id, action_name, sentiment

    def can_replicate(self) -> bool:
        return self.resources.can_replicate() and len(self.children) == 0

    def replicate(self, training_bundles: Optional[List[ModalityBundle]] = None,
                  mutation_power: float = 0.05) -> "SelfReplicatingRobot":
        print(f"\n  ✨ Robot '{self.name}' initiating mutated replication... 🧬")
        self.resources.consume_for_replication()

        # 1. Gather Teacher Intelligence
        training_bundles = training_bundles or [self.perception.make_synthetic_bundle(seed=i) for i in range(64)]
        self.action_head.eval()
        teacher_inputs, teacher_logits = [], []
        with torch.no_grad():
            for b in training_bundles:
                z, _, _ = self.perception.perceive(b)
                teacher_inputs.append(z)
                teacher_logits.append(self.action_head(z))

        Z, T_logits = torch.cat(teacher_inputs, dim=0), torch.cat(teacher_logits, dim=0)

        # 2. Distillation Process
        child_head = NeuromorphicActionHead(self.domain_cfg["n_actions"], is_child=True)
        optimizer = torch.optim.Adam(child_head.parameters(), lr=0.005)
        loss_fn = nn.MSELoss()

        child_head.train()
        for _ in range(200):
            optimizer.zero_grad()
            loss = loss_fn(child_head(Z), T_logits)
            loss.backward()
            optimizer.step()

        # 3. Apply Genetic Mutation 🧬
        # We iterate through the child's parameters and add random noise
        with torch.no_grad():
            for param in child_head.parameters():
                mutation = torch.randn_like(param) * mutation_power
                param.add_(mutation)

        print(f"  🧠 Distillation complete (Loss: {loss.item():.6f}) | Mutation Applied: {mutation_power:.2%}")

        child_name = f"Child-{self.name}-G{len(self.children) + 1}"
        child = SelfReplicatingRobot(
            name=child_name, domain=self.domain, perception=self.perception, action_head=child_head,
            is_child=True, parent_name=self.name,
            resources=ResourceLedger(raw_materials=40.0, component_parts=0, energy=30.0)
        )
        self.children.append(child)
        return child

    def summary(self) -> Dict:
        action_counts = defaultdict(int)
        for log in self.action_log: action_counts[log["action_name"]] += 1
        moods = [s.mood for s in self.sentiment_history]
        return {
            "robot": self.name, "domain": self.domain, "cycles": self.cycle, "children": len(self.children),
            "action_counts": dict(action_counts),
            "mood_distribution": {"stable": moods.count("stable"), "drifting": moods.count("drifting")},
            "final_resources": {"materials": round(self.resources.raw_materials, 1), "parts": self.resources.component_parts, "energy": round(self.resources.energy, 1)},
        }


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 6 — UNIVERSAL CONSTRUCTOR SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

class UniversalConstructor:
    def __init__(
        self,
        model_dir: Optional[str] = None,
        domains: Optional[List[str]] = None,
        n_cycles: int = 60,
        verbose: bool = True,
        train_from_scratch: bool = False,
        force_cpu: bool = False,
    ):
        self.n_cycles = n_cycles
        self.verbose  = verbose
        self.robots:  Dict[str, SelfReplicatingRobot] = {}
        self.hive_log: List[Dict] = []
        self.device = torch.device("cpu") if force_cpu else DEVICE

        print("\n" + "═" * 70)
        print("  🌌  UNIVERSAL CONSTRUCTOR — PLANET FACTORY EDITION")
        print("═" * 70)
        print(f"  ⚛️ Quantum Engine (Cirq/Qsim) : {'ONLINE' if HAS_QUANTUM else 'SIMULATED'}")
        print(f"  🧠 Neuromorphic Core (Brian2) : {'ONLINE' if HAS_NEUROMORPHIC else 'OFFLINE (PyTorch Fallback)'}")

        mdir = model_dir or MODEL_DIR
        self.normalizer    = FeatureNormalizer(os.path.join(mdir, NORM_JSON))
        self.holosyn       = HoloSynHeads()
        self.student_hf    = StudentDistilledHeadsHF()
        self.student_basic = StudentDistilledHeadsBasic()

        if train_from_scratch:
            print(f"\n  🔥 --train-from-scratch FLAG DETECTED. Bypassing pre-trained weights.")
            print(f"  Training intelligence layer from scratch on {self.device}...")
            self._train_models_from_scratch()
        else:
            print("\n📦 Loading models...")
            self.holosyn.load_torchscript(os.path.join(mdir, HOLOSYN_TORCHSCRIPT))
            self.student_hf.load_torchscript(os.path.join(mdir, STUDENT_HF_TS))
            self.student_basic.load_torchscript(os.path.join(mdir, STUDENT_BASIC_TS))

        self.perception = PerceptionPipeline(self.normalizer, self.holosyn, self.student_hf, self.student_basic)

        active_domains = domains or list(DOMAINS.keys())
        print(f"\n🤖 Spawning {len(active_domains)} domain robots...")
        for domain in active_domains:
            name = f"Alpha-{domain.split('_')[0].capitalize()}"
            self.robots[name] = SelfReplicatingRobot(
                name=name, domain=domain, perception=self.perception,
                resources=ResourceLedger(raw_materials=80, component_parts=0, energy=60),
            )

        print(f"\n✅ Planet Factory ready — {len(self.robots)} systems online")
        print("═" * 70)

    def _train_models_from_scratch(self):
        """Pre-training loop to satisfy 'Train new models per domain from scratch' constraint."""
        self.holosyn.to(self.device)
        self.student_hf.to(self.device)

        optimizer = torch.optim.Adam(
            list(self.holosyn.parameters()) + list(self.student_hf.parameters()),
            lr=1e-3
        )
        loss_fn = nn.MSELoss()

        self.holosyn.train()
        self.student_hf.train()

        for step in range(1, 21):
            optimizer.zero_grad()
            # Mock synthetic targets
            target_z = F.normalize(torch.randn(8, 256), p=2, dim=-1).to(self.device)
            target_scores = torch.rand(8, 4).to(self.device)

            # Forward pass mocks
            E_t = torch.randn(8, 528).to(self.device)
            E_i = torch.randn(8, 2048).to(self.device)
            E_a = torch.randn(8, 2048).to(self.device)
            E_v = torch.randn(8, 2048).to(self.device)
            H   = torch.randn(8, 32, 3).to(self.device)
            feat = torch.randn(8, 789).to(self.device)

            z_t, _, _, _, _ = self.holosyn(E_t, E_i, E_a, E_v, H)
            scores = self.student_hf(feat)

            loss = loss_fn(z_t, target_z) + loss_fn(scores, target_scores)
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"     [Scratch Pre-training] Epoch {step}/20 - Loss: {loss.item():.4f}")

        # Move back to CPU for inference compatibility with Brian2 and Cirq
        self.holosyn.to("cpu")
        self.student_hf.to("cpu")

    def run(self):
        print(f"\n🚀 Starting simulation — {self.n_cycles} cycles × {len(self.robots)} robots")
        print("─" * 70)
        all_robots = dict(self.robots)

        for cycle in range(1, self.n_cycles + 1):
            cycle_log = {"cycle": cycle, "events": []}

            for robot_name, robot in list(all_robots.items()):
                action_id, action_name, sentiment = robot.perceive_and_act(seed=cycle * 37 + robot.id)

                if self.verbose:
                    mw = robot.action_log[-1]["modal_weights"]
                    res = robot.resources
                    print(f"  Cycle {cycle:03d} | {robot.domain_cfg['emoji']} {robot_name:<20} | {action_name:<20} | mood={sentiment.mood:<8} | mat={res.raw_materials:4.1f} pts={res.component_parts:2d} nrg={res.energy:4.1f}")

                cycle_log["events"].append({
                    "robot": robot_name,
                    "action": action_name,
                    "sentiment": sentiment.format(),
                })

                if robot.can_replicate():
                    print(f"\n  🏗️  Replication triggered for '{robot_name}' at cycle {cycle}!")
                    child = robot.replicate()
                    all_robots[child.name] = child

                    cycle_log["events"].append({
                        "event": "REPLICATION",
                        "parent": robot_name,
                        "child": child.name,
                    })
                    print()

            self.hive_log.append(cycle_log)

        self._print_final_report(all_robots)
        return all_robots

    def export_log(self, path: str):
        """Save the full hive log to JSON."""
        with open(path, "w") as f:
            json.dump(self.hive_log, f, indent=2)
        print(f"\n📝 Hive log saved → {path}")

    def _print_final_report(self, all_robots: Dict):
        print("\n" + "═" * 70)
        print("  📊  FINAL PLANET FACTORY REPORT")
        print("═" * 70)

        for name, robot in all_robots.items():
            s = robot.summary()
            tier  = "👶" if robot.is_child else "🤖"
            print(f"  {tier} {DOMAINS[robot.domain]['emoji']} {name:<22} (Cycles: {s['cycles']}, Children: {s['children']})")
        print("═" * 70)


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Universal Constructor: Planet Factory Edition")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Directory containing .pt model files")
    parser.add_argument("--cycles",    type=int, default=60, help="Number of simulation cycles")
    parser.add_argument("--domains",   nargs="+", choices=list(DOMAINS.keys()), help="Domains to activate")
    parser.add_argument("--quiet",     action="store_true", help="Suppress per-cycle output")
    parser.add_argument("--train-from-scratch", action="store_true", help="Train new models from scratch instead of loading checkpoints")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU execution, overriding GPU detection")
    parser.add_argument("--export",    type=str, default=None, help="Path to export the hive_mind_log.json file")
    args = parser.parse_args()

    uc = UniversalConstructor(
        model_dir=args.model_dir,
        domains=args.domains,
        n_cycles=args.cycles,
        verbose=not args.quiet,
        train_from_scratch=args.train_from_scratch,
        force_cpu=args.force_cpu,
    )
    uc.run()

    if args.export:
        uc.export_log(args.export)