"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              UNIVERSAL CONSTRUCTOR — ROBOT FRAMEWORK v2.0                  ║
║                                                                              ║
║  Rebuilt using production-trained models:                                    ║
║   • HoloSynHeads          — multimodal teacher projectors                   ║
║   • StudentDistilledHeads — learned modality-scoring gate (HF + basic)      ║
║   • student_norm_hf.json  — feature normalization statistics                ║
║                                                                              ║
║  Architecture:                                                               ║
║   Sensor Input (text/img/audio/video/haptics)                                ║
║        ↓  Feature Extraction + Normalization                                 ║
║   HoloSynHeads (Teacher)  →  5 × 256-dim L2-normalized embeddings           ║
║   StudentDistilledHeads   →  4-dim modality attention weights (Sigmoid)      ║
║        ↓  Masked weighted fusion                                             ║
║   256-dim fused state embedding                                              ║
║        ↓  Domain action head                                                 ║
║   Robot action + Quantum Sentiment                                           ║
║        ↓  Resource economics                                                 ║
║   Self-replicating child via knowledge distillation                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
import math
import warnings
import zipfile
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS  (Updated to match the newly uploaded model files)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

HOLOSYN_TORCHSCRIPT = os.path.join(MODEL_DIR, "holosyn_heads.torchscript.pt")
HOLOSYN_STATEDICT = os.path.join(MODEL_DIR, "holosyn_heads.pt")
STUDENT_HF_TS = os.path.join(MODEL_DIR, "student_distilled_heads_hf.torchscript.pt")
STUDENT_BASIC_TS = os.path.join(MODEL_DIR, "student_distilled_heads.torchscript.pt")
NORM_JSON = os.path.join(MODEL_DIR, "student_norm_hf.json")


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 0 — FEATURE NORMALIZATION
# ═════════════════════════════════════════════════════════════════════════════

class FeatureNormalizer:
    """
    Applies z-score normalization from student_norm_hf.json.

    Columns (789 total):
      [0..4]   txt_len, txt_lines, txt_exclaim, txt_question, txt_caps_ratio
      [5..7]   hapt_len, hapt_has_intensity, hapt_has_freq
      [8..11]  aud_rms, aud_zcr, aud_centroid, aud_tempo
      [12..20] img_w, img_h, img_mean_rgb(3), img_std_rgb(3), vid_n_frames
      [21..788] w2v_0 … w2v_767
    """

    # Indices for each feature group
    TXT_SLICE = slice(0, 5)
    HAPT_SLICE = slice(5, 8)
    AUD_SLICE = slice(8, 12)
    IMG_SLICE = slice(12, 21)
    W2V_SLICE = slice(21, 789)
    FULL_DIM = 789

    def __init__(self, norm_path: Optional[str] = None):
        self.mu = np.zeros(self.FULL_DIM, dtype=np.float32)
        self.sd = np.ones(self.FULL_DIM, dtype=np.float32)
        self.columns: List[str] = []
        if norm_path and os.path.exists(norm_path):
            self._load(norm_path)
        else:
            print(f"  ⚠️ Normalization stats JSON not found at {norm_path}. Using identity normalization (mu=0, sd=1).")

    def _load(self, path: str):
        with open(path) as f:
            d = json.load(f)
        self.columns = d["numeric_cols"]
        self.mu = np.array(d["mu"], dtype=np.float32)
        self.sd = np.array(d["sd"], dtype=np.float32)
        self.sd = np.where(self.sd < 1e-9, 1.0, self.sd)  # guard zero-sd cols
        print(f"  ✅ Normalization stats loaded: {len(self.columns)} features")

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """x: (..., 789) → normalized float32 array"""
        return (x - self.mu) / self.sd

    def make_feature_vector(
            self,
            txt_meta: Optional[np.ndarray] = None,  # (5,)
            hapt_meta: Optional[np.ndarray] = None,  # (3,)
            aud_meta: Optional[np.ndarray] = None,  # (4,)
            img_meta: Optional[np.ndarray] = None,  # (9,)
            w2v: Optional[np.ndarray] = None,  # (768,)
    ) -> np.ndarray:
        """Assemble and normalise a full 789-dim feature vector."""
        vec = np.zeros(self.FULL_DIM, dtype=np.float32)
        if txt_meta is not None: vec[self.TXT_SLICE] = txt_meta[:5]
        if hapt_meta is not None: vec[self.HAPT_SLICE] = hapt_meta[:3]
        if aud_meta is not None: vec[self.AUD_SLICE] = aud_meta[:4]
        if img_meta is not None: vec[self.IMG_SLICE] = img_meta[:9]
        if w2v is not None: vec[self.W2V_SLICE] = w2v[:768]
        return self.normalize(vec)


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 1 — MODEL DEFINITIONS (matching real checkpoint dims exactly)
# ═════════════════════════════════════════════════════════════════════════════

class Projector(nn.Module):
    """
    MLP projector with L2-normalised output.
    Architecture: Linear(in_dim, hidden) → ReLU → Linear(hidden, out_dim) → L2-norm
    """

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
    """
    1-D CNN encoder for haptic signal sequences.
    Input H: (B, T, 3)  — T time steps, 3 haptic channels (fx, fy, fz)
    Output:  (B, 256)   — L2-normalised haptic embedding
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(64, 256)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: (B, T, 3) → transpose → (B, 3, T) for Conv1d
        x = H.transpose(1, 2)
        x = self.net(x).squeeze(-1)  # (B, 64)
        x = self.proj(x)  # (B, 256)
        return F.normalize(x, p=2, dim=-1)


class HoloSynHeads(nn.Module):
    """
    Teacher multimodal projector ensemble.

    Input dimensions (from real checkpoint):
      E_t_aug: (B, 528)   — text embedding + metadata augmentation
      E_i:     (B, 2048)  — image embedding (e.g. ResNet-50)
      E_a:     (B, 2048)  — audio embedding
      E_v:     (B, 2048)  — video embedding
      H:       (B, T, 3)  — haptic time series

    Output: 5 × (B, 256) L2-normalised embeddings, one per modality
    """
    TEXT_DIM = 528
    IMG_DIM = 2048
    AUD_DIM = 2048
    VID_DIM = 2048

    def __init__(self):
        super().__init__()
        self.p_t = Projector(self.TEXT_DIM, hidden=512, out_dim=256)
        self.p_i = Projector(self.IMG_DIM, hidden=512, out_dim=256)
        self.p_a = Projector(self.AUD_DIM, hidden=512, out_dim=256)
        self.p_v = Projector(self.VID_DIM, hidden=512, out_dim=256)
        self.h_enc = HapticsEncoder()

    def forward(
            self,
            E_t_aug: torch.Tensor,
            E_i: torch.Tensor,
            E_a: torch.Tensor,
            E_v: torch.Tensor,
            H: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        return (
            self.p_t(E_t_aug),
            self.p_i(E_i),
            self.p_a(E_a),
            self.p_v(E_v),
            self.h_enc(H),
        )

    def load_torchscript(self, path: str) -> bool:
        """Load weights from TorchScript checkpoint (no torch.jit required)."""
        if not os.path.exists(path):
            print(f"  ⚠️  HoloSynHeads: File not found at {path}. Using random init.")
            return False
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                self.load_state_dict(ckpt, strict=False)
            print(f"  ✅ HoloSynHeads weights loaded from {os.path.basename(path)}")
            return True
        except Exception as e:
            print(f"  ⚠️  HoloSynHeads: could not load weights ({e}). Using random init.")
            return False


class StudentDistilledHeadsHF(nn.Module):
    """
    Full-feature modality-scoring gate (HuggingFace / production variant).

    Input:  (B, 789)  — normalised feature vector from FeatureNormalizer
    Output: (B, 4)    — per-modality confidence scores ∈ [0,1]
                         order: [text, image, audio, video]

    Architecture: Linear(789,256)→Tanh→Linear(256,128)→Tanh→Linear(128,4)→Sigmoid
    """
    IN_DIM = 789

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(789, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 4), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def load_torchscript(self, path: str) -> bool:
        if not os.path.exists(path):
            print(f"  ⚠️  StudentDistilledHeadsHF: File not found at {path}. Using random init.")
            return False
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                self.load_state_dict(ckpt, strict=False)
            print(f"  ✅ StudentDistilledHeadsHF loaded from {os.path.basename(path)}")
            return True
        except Exception as e:
            print(f"  ⚠️  StudentDistilledHeadsHF: could not load ({e}). Using random init.")
            return False


class StudentDistilledHeadsBasic(nn.Module):
    """
    Lightweight modality-scoring gate (metadata-only, no W2V).

    Input:  (B, 5)  — [txt_len, txt_lines, txt_exclaim, txt_question, txt_caps_ratio]
    Output: (B, 4)  — per-modality confidence scores ∈ [0,1]

    Architecture: Linear(5,64)→Tanh→Linear(64,32)→Tanh→Linear(32,4)→Sigmoid
    """
    IN_DIM = 5

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
        if not os.path.exists(path):
            print(f"  ⚠️  StudentDistilledHeadsBasic: File not found at {path}. Using random init.")
            return False
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                self.load_state_dict(ckpt, strict=False)
            print(f"  ✅ StudentDistilledHeadsBasic loaded from {os.path.basename(path)}")
            return True
        except Exception as e:
            print(f"  ⚠️  StudentDistilledHeadsBasic: could not load ({e}). Using random init.")
            return False


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — MULTIMODAL PERCEPTION PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ModalityBundle:
    """Container for one observation across all modalities."""
    E_t_aug: torch.Tensor  # (1, 528) text+meta
    E_i: torch.Tensor  # (1, 2048) image
    E_a: torch.Tensor  # (1, 2048) audio
    E_v: torch.Tensor  # (1, 2048) video
    H: torch.Tensor  # (1, T, 3) haptics
    feat_hf: torch.Tensor  # (1, 789) norm features for HF gate
    feat_basic: torch.Tensor  # (1, 5)   txt metadata for basic gate
    mask: torch.Tensor  # (1, 4) which modalities are present
    meta: Dict = field(default_factory=dict)


class PerceptionPipeline:
    """
    Converts raw sensor data into a fused 256-dim state embedding.

    Fusion:
      1. HoloSynHeads projects each modality to 256-dim L2-normalised space.
      2. StudentDistilledHeadsHF scores each modality given the full feature vec.
      3. Attention weights (masked softmax) combine the 4 text/img/aud/vid embeddings.
      4. Output is L2-normalised.
    """

    def __init__(self, normalizer: FeatureNormalizer,
                 holosyn: HoloSynHeads,
                 student_hf: StudentDistilledHeadsHF,
                 student_basic: StudentDistilledHeadsBasic):
        self.norm = normalizer
        self.holosyn = holosyn
        self.student_hf = student_hf
        self.student_basic = student_basic
        self.holosyn.eval()
        self.student_hf.eval()
        self.student_basic.eval()

    @torch.no_grad()
    def perceive(self, bundle: ModalityBundle) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Returns:
          fused_emb: (1, 256) — weighted fused state embedding
          modal_weights: (1, 4) — softmax attention over [text, img, aud, vid]
          info: dict with per-modality embeddings and scores
        """
        # 1. Project each modality via teacher heads
        z_t, z_i, z_a, z_v, z_h = self.holosyn(
            bundle.E_t_aug, bundle.E_i, bundle.E_a, bundle.E_v, bundle.H
        )

        # 2. Score modalities with student gate (HF variant if features available)
        if bundle.feat_hf is not None:
            raw_scores = self.student_hf(bundle.feat_hf)  # (1,4) ∈ [0,1]
        else:
            raw_scores = self.student_basic(bundle.feat_basic)  # (1,4) ∈ [0,1]

        # 3. Mask absent modalities then softmax
        masked = raw_scores * bundle.mask  # zero-out absent
        # Avoid -inf when all masked → fall back to uniform
        logits = torch.where(bundle.mask > 0,
                             torch.log(masked + 1e-9),
                             torch.full_like(masked, -1e9))
        modal_weights = torch.softmax(logits, dim=-1)  # (1,4)

        # 4. Weighted sum of modal embeddings
        w = modal_weights.unsqueeze(-1)  # (1,4,1)
        stack = torch.stack([z_t, z_i, z_a, z_v], dim=1)  # (1,4,256)
        fused = (w * stack).sum(dim=1)  # (1,256)
        fused = F.normalize(fused, p=2, dim=-1)

        info = {
            "z_text": z_t, "z_img": z_i, "z_aud": z_a, "z_vid": z_v,
            "z_hapt": z_h, "modal_scores": raw_scores,
            "modal_weights": modal_weights,
        }
        return fused, modal_weights, info

    def make_synthetic_bundle(
            self,
            txt_meta: Optional[np.ndarray] = None,
            haptic_seq: Optional[np.ndarray] = None,
            available_modalities: List[str] = None,
            seed: Optional[int] = None,
    ) -> ModalityBundle:
        """
        Create a synthetic ModalityBundle for testing / simulation without
        real sensor hardware. All embeddings are randomly sampled.
        """
        rng = np.random.default_rng(seed)
        available = set(available_modalities or ["text", "image", "audio", "video"])

        # Text metadata
        if txt_meta is None:
            txt_meta = np.array([
                rng.integers(10, 200),  # txt_len
                rng.integers(1, 10),  # txt_lines
                rng.integers(0, 5),  # txt_exclaim
                rng.integers(0, 3),  # txt_question
                rng.uniform(0, 0.15),  # txt_caps_ratio
            ], dtype=np.float32)

        # Random embeddings matching real model input dims
        E_t_aug_np = rng.standard_normal(528).astype(np.float32)
        E_i_np = rng.standard_normal(2048).astype(np.float32)
        E_a_np = rng.standard_normal(2048).astype(np.float32)
        E_v_np = rng.standard_normal(2048).astype(np.float32)

        # Haptics: (T=32, 3)
        if haptic_seq is None:
            t = np.linspace(0, 2 * np.pi, 32)
            haptic_seq = np.stack([
                np.sin(t) + rng.standard_normal(32) * 0.1,
                np.cos(t) + rng.standard_normal(32) * 0.1,
                0.5 * np.sin(2 * t) + rng.standard_normal(32) * 0.05,
            ], axis=-1).astype(np.float32)  # (32, 3)

        # Full feature vector (789-dim)
        w2v = rng.standard_normal(768).astype(np.float32)
        aud_meta = np.array([
            rng.uniform(0.0, 0.5),  # aud_rms
            rng.uniform(0.0, 0.05),  # aud_zcr
            rng.uniform(100, 4000),  # aud_centroid
            rng.uniform(60, 180),  # aud_tempo
        ], dtype=np.float32)
        img_meta = np.zeros(9, dtype=np.float32)

        feat_789 = self.norm.make_feature_vector(
            txt_meta=txt_meta, aud_meta=aud_meta,
            img_meta=img_meta, w2v=w2v
        )

        # Modality availability mask [text, img, aud, vid]
        mask = np.array([
            1.0 if "text" in available else 0.0,
            1.0 if "image" in available else 0.0,
            1.0 if "audio" in available else 0.0,
            1.0 if "video" in available else 0.0,
        ], dtype=np.float32)

        return ModalityBundle(
            E_t_aug=torch.from_numpy(E_t_aug_np).unsqueeze(0),
            E_i=torch.from_numpy(E_i_np).unsqueeze(0),
            E_a=torch.from_numpy(E_a_np).unsqueeze(0),
            E_v=torch.from_numpy(E_v_np).unsqueeze(0),
            H=torch.from_numpy(haptic_seq).unsqueeze(0),
            feat_hf=torch.from_numpy(feat_789).unsqueeze(0),
            feat_basic=torch.from_numpy(txt_meta[:5]).unsqueeze(0),
            mask=torch.from_numpy(mask).unsqueeze(0),
            meta={"txt_meta": txt_meta, "aud_meta": aud_meta},
        )


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 3 — QUANTUM SENTIMENT ENGINE
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class QuantumSentiment:
    """
    Simulates classical + quantum emotional state metrics per robot agent.
    H = Harmony, M = Magnitude, S = Synchrony
    Classical values ∈ [0,1] using softmax over cosine-sim clusters.
    Quantum values ∈ [0,1] via Born-rule probability simulation.
    """
    cycle: int
    agent: str
    classical_H: float
    classical_M: float
    classical_S: float
    quantum_H: float
    quantum_M: float
    quantum_S: float
    mood: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @staticmethod
    def from_embedding(emb: torch.Tensor, agent: str, cycle: int) -> "QuantumSentiment":
        """Derive sentiment from a 256-dim fused embedding."""
        v = emb.squeeze().detach().numpy()

        # Classical: sliding-window statistics on embedding quartiles
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        c_H = float(np.clip((q2 + 1) / 2, 0, 1))
        c_M = float(np.clip(np.linalg.norm(v) / 16, 0, 1))
        c_S = float(np.clip((q3 - q1 + 1) / 2, 0, 1))

        # Quantum: Born-rule simulation via squared amplitude
        probs = np.abs(v[:8]) ** 2
        probs /= probs.sum() + 1e-9
        q_H = float(probs[:3].sum())
        q_M = float(probs[3:6].sum())
        q_S = float(probs[6:].sum())

        # Mood classification
        stability = abs(c_H - 0.5) + abs(c_M - 0.5)
        mood = "stable" if stability < 0.25 else "drifting"

        return QuantumSentiment(
            cycle=cycle, agent=agent,
            classical_H=round(c_H, 3), classical_M=round(c_M, 3), classical_S=round(c_S, 3),
            quantum_H=round(q_H, 3), quantum_M=round(q_M, 3), quantum_S=round(q_S, 3),
            mood=mood,
        )

    def format(self) -> str:
        return (
            f"[{self.agent}] cycle={self.cycle} mood={self.mood}. "
            f"classical(H={self.classical_H}, M={self.classical_M}, S={self.classical_S}) "
            f"quantum(H={self.quantum_H}, M={self.quantum_M}, S={self.quantum_S})."
        )


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 4 — DOMAIN ACTION HEADS
# ═════════════════════════════════════════════════════════════════════════════

class DomainActionHead(nn.Module):
    """
    Lightweight action head that maps fused embedding → discrete action.
    Parent: 256→128→64→n_actions
    Child (distilled): 256→32→n_actions
    """

    def __init__(self, n_actions: int, is_child: bool = False):
        super().__init__()
        self.is_child = is_child
        if is_child:
            self.net = nn.Sequential(
                nn.Linear(256, 32), nn.ReLU(),
                nn.Linear(32, n_actions),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, n_actions),
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    @torch.no_grad()
    def predict(self, z: torch.Tensor) -> int:
        self.eval()
        logits = self(z)
        return int(logits.argmax(dim=-1).item())


DOMAINS: Dict[str, Dict] = {
    "robot_arm": {
        "description": "DROID Robot Arm Manipulation",
        "n_actions": 2,
        "action_map": {0: "➡️  Move arm", 1: "🤝 Grasp object"},
        "emoji": "🦾",
    },
    "self_driving": {
        "description": "Autonomous Navigation",
        "n_actions": 3,
        "action_map": {0: "⬅️  Steer left", 1: "⬆️  Go straight", 2: "➡️  Steer right"},
        "emoji": "🚗",
    },
    "asteroid_factory": {
        "description": "Asteroid Mining & Factory",
        "n_actions": 2,
        "action_map": {0: "🌿 Conserve energy", 1: "⛏️  Mine resources"},
        "emoji": "☄️",
    },
    "cnc_production": {
        "description": "CNC Precision Manufacturing",
        "n_actions": 2,
        "action_map": {0: "✅ Maintain params", 1: "🔧 Adjust parameters"},
        "emoji": "⚙️",
    },
    "building_automation": {
        "description": "Structural Building Automation",
        "n_actions": 4,
        "action_map": {0: "👁️  Monitor", 1: "🧱 Reinforce", 2: "🔥 Weld", 3: "🔩 Assemble"},
        "emoji": "🏗️",
    },
    "universe_builder": {
        "description": "Cosmic Factory Constructor",
        "n_actions": 3,
        "action_map": {0: "🌑 Asteroid factory", 1: "☄️  Comet factory", 2: "🌍 Planet factory"},
        "emoji": "🌌",
    },
}


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 5 — SELF-REPLICATING ROBOT
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ResourceLedger:
    """Economic accounting for self-replication."""
    raw_materials: float = 100.0
    component_parts: int = 0
    energy: float = 50.0
    replication_cost_materials: float = 20.0
    replication_cost_parts: int = 10
    replication_cost_energy: float = 15.0

    def can_replicate(self) -> bool:
        return (self.raw_materials >= self.replication_cost_materials and
                self.component_parts >= self.replication_cost_parts and
                self.energy >= self.replication_cost_energy)

    def consume_for_replication(self):
        self.raw_materials -= self.replication_cost_materials
        self.component_parts -= self.replication_cost_parts
        self.energy -= self.replication_cost_energy

    def update_for_action(self, action_id: int, domain: str):
        """Simulate resource changes based on the chosen action."""
        # Generic resource economics: productive actions consume materials,
        # produce parts; conservative actions restore energy
        if domain in ("asteroid_factory", "cnc_production"):
            if action_id == 1:  # productive
                self.raw_materials = max(0, self.raw_materials - 2)
                self.component_parts += 1
                self.energy = max(0, self.energy - 1)
            else:  # conservative
                self.energy = min(100, self.energy + 0.5)
        elif domain == "building_automation":
            if action_id in (1, 2, 3):
                self.raw_materials = max(0, self.raw_materials - 3)
                self.component_parts += 1
                self.energy = max(0, self.energy - 2)
        elif domain == "universe_builder":
            self.raw_materials = max(0, self.raw_materials - 5)
            self.component_parts += 2
            self.energy = max(0, self.energy - 3)
        else:
            # robot_arm, self_driving
            self.energy = max(0, self.energy - 0.3)


class SelfReplicatingRobot:
    """
    A fully-embodied robot agent with:
      • Multimodal perception via HoloSynHeads + StudentDistilledHeads
      • Domain-specific action selection via DomainActionHead
      • Quantum sentiment tracking
      • Resource-gated self-replication via knowledge distillation
    """

    _robot_count = 0

    def __init__(
            self,
            name: str,
            domain: str,
            perception: PerceptionPipeline,
            action_head: Optional[DomainActionHead] = None,
            is_child: bool = False,
            parent_name: Optional[str] = None,
            resources: Optional[ResourceLedger] = None,
    ):
        SelfReplicatingRobot._robot_count += 1
        self.id = SelfReplicatingRobot._robot_count
        self.name = name
        self.domain = domain
        self.domain_cfg = DOMAINS[domain]
        self.perception = perception
        self.is_child = is_child
        self.parent_name = parent_name
        self.resources = resources or ResourceLedger()
        self.children: List["SelfReplicatingRobot"] = []
        self.sentiment_history: List[QuantumSentiment] = []
        self.action_log: List[Dict] = []
        self.cycle = 0

        n_actions = self.domain_cfg["n_actions"]
        self.action_head = action_head or DomainActionHead(n_actions, is_child=is_child)

        lineage = f" (child of {parent_name})" if parent_name else ""
        tier = "👶 Child" if is_child else "🤖 Parent"
        print(f"  {tier} Robot '{name}'{lineage} — domain: {self.domain_cfg['emoji']} {domain}")

    # ── Perception ────────────────────────────────────────────────────────

    def perceive_and_act(
            self,
            bundle: Optional[ModalityBundle] = None,
            seed: Optional[int] = None,
    ) -> Tuple[int, str, QuantumSentiment]:
        """
        Full perception→action cycle:
          1. Generate or use provided ModalityBundle
          2. Fuse modalities via HoloSynHeads + StudentDistilledHeads
          3. Select action via domain head
          4. Compute quantum sentiment from fused embedding
        """
        self.cycle += 1

        if bundle is None:
            bundle = self.perception.make_synthetic_bundle(seed=seed)

        # Perception
        fused_emb, modal_weights, info = self.perception.perceive(bundle)

        # Action selection
        with torch.no_grad():
            action_id = self.action_head.predict(fused_emb)

        action_name = self.domain_cfg["action_map"].get(action_id, "Unknown")

        # Quantum sentiment
        sentiment = QuantumSentiment.from_embedding(fused_emb, self.name, self.cycle)
        self.sentiment_history.append(sentiment)

        # Resource update
        self.resources.update_for_action(action_id, self.domain)

        # Log
        self.action_log.append({
            "cycle": self.cycle,
            "action_id": action_id,
            "action_name": action_name,
            "modal_weights": modal_weights.squeeze().tolist(),
            "mood": sentiment.mood,
            "resources": {
                "materials": round(self.resources.raw_materials, 1),
                "parts": self.resources.component_parts,
                "energy": round(self.resources.energy, 1),
            }
        })

        return action_id, action_name, sentiment

    # ── Replication ───────────────────────────────────────────────────────

    def can_replicate(self) -> bool:
        return self.resources.can_replicate() and len(self.children) == 0

    def replicate(
            self,
            training_bundles: Optional[List[ModalityBundle]] = None,
            n_distill_steps: int = 200,
            lr: float = 0.005,
    ) -> "SelfReplicatingRobot":
        """
        Spawn a child robot via knowledge distillation:
          1. Collect teacher logits from parent's action head
          2. Train a smaller child head to match them (MSE)
          3. Wrap in a new SelfReplicatingRobot with reduced resources
        """
        print(f"\n  ✨ Robot '{self.name}' initiating replication... 🧬")

        # Consume resources
        self.resources.consume_for_replication()

        # Generate training data if not provided
        if training_bundles is None:
            training_bundles = [
                self.perception.make_synthetic_bundle(seed=i)
                for i in range(64)
            ]

        # Collect teacher embeddings and logits
        self.action_head.eval()
        teacher_inputs, teacher_logits = [], []
        with torch.no_grad():
            for b in training_bundles:
                z, _, _ = self.perception.perceive(b)
                teacher_inputs.append(z)
                teacher_logits.append(self.action_head(z))
        Z = torch.cat(teacher_inputs, dim=0)  # (N, 256)
        T_logits = torch.cat(teacher_logits, dim=0)  # (N, n_actions)

        # Train child head
        n_actions = self.domain_cfg["n_actions"]
        child_head = DomainActionHead(n_actions, is_child=True)
        optimizer = torch.optim.Adam(child_head.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        child_head.train()
        for step in range(n_distill_steps):
            pred = child_head(Z)
            loss = loss_fn(pred, T_logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        print(f"  🧠 Distillation complete — final loss: {final_loss:.6f}")

        child_name = f"Child-{self.name}-G{len(self.children) + 1}"
        child = SelfReplicatingRobot(
            name=child_name,
            domain=self.domain,
            perception=self.perception,  # shared perception stack
            action_head=child_head,
            is_child=True,
            parent_name=self.name,
            resources=ResourceLedger(
                raw_materials=40.0,
                component_parts=0,
                energy=30.0,
            ),
        )
        self.children.append(child)
        print(f"  ✅ Child robot '{child_name}' is live and inherits parent's policy.")
        return child

    # ── Reporting ─────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        action_counts = defaultdict(int)
        for log in self.action_log:
            action_counts[log["action_name"]] += 1
        moods = [s.mood for s in self.sentiment_history]
        return {
            "robot": self.name,
            "domain": self.domain,
            "cycles": self.cycle,
            "is_child": self.is_child,
            "children": len(self.children),
            "action_counts": dict(action_counts),
            "mood_distribution": {
                "stable": moods.count("stable"),
                "drifting": moods.count("drifting"),
            },
            "final_resources": {
                "materials": round(self.resources.raw_materials, 1),
                "parts": self.resources.component_parts,
                "energy": round(self.resources.energy, 1),
            },
        }


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 6 — UNIVERSAL CONSTRUCTOR SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

class UniversalConstructor:
    """
    Top-level simulation engine.

    Manages a hive of SelfReplicatingRobots across multiple domains,
    coordinated through the shared HoloSynHeads perception pipeline.

    Key features:
      • Multi-domain simultaneous operation
      • Automatic replication when resources permit
      • Quantum sentiment hive mind
      • Structured reporting
    """

    def __init__(
            self,
            model_dir: Optional[str] = None,
            domains: Optional[List[str]] = None,
            n_cycles: int = 60,
            verbose: bool = True,
    ):
        self.n_cycles = n_cycles
        self.verbose = verbose
        self.robots: Dict[str, SelfReplicatingRobot] = {}
        self.hive_log: List[Dict] = []

        mdir = model_dir or MODEL_DIR
        print("\n" + "═" * 70)
        print("  🌌  UNIVERSAL CONSTRUCTOR — INITIALIZING")
        print("═" * 70)

        # ── Load models ──────────────────────────────────────────────────
        print("\n📦 Loading models...")
        self.normalizer = FeatureNormalizer(os.path.join(mdir, NORM_JSON))
        self.holosyn = HoloSynHeads()
        self.student_hf = StudentDistilledHeadsHF()
        self.student_basic = StudentDistilledHeadsBasic()

        hs_path = os.path.join(mdir, HOLOSYN_TORCHSCRIPT)
        hf_path = os.path.join(mdir, STUDENT_HF_TS)
        bs_path = os.path.join(mdir, STUDENT_BASIC_TS)

        self.holosyn.load_torchscript(hs_path)
        self.student_hf.load_torchscript(hf_path)
        self.student_basic.load_torchscript(bs_path)

        # ── Build shared perception pipeline ─────────────────────────────
        self.perception = PerceptionPipeline(
            self.normalizer, self.holosyn,
            self.student_hf, self.student_basic,
        )

        # ── Spawn initial robots ──────────────────────────────────────────
        active_domains = domains or list(DOMAINS.keys())
        print(f"\n🤖 Spawning {len(active_domains)} domain robots...")
        for domain in active_domains:
            cfg = DOMAINS[domain]
            name = f"Alpha-{domain.split('_')[0].capitalize()}"
            robot = SelfReplicatingRobot(
                name=name, domain=domain,
                perception=self.perception,
                resources=ResourceLedger(raw_materials=60, component_parts=0, energy=50),
            )
            self.robots[name] = robot

        print(f"\n✅ Universal Constructor ready — {len(self.robots)} robots online")
        print("═" * 70)

    # ── Main simulation loop ───────────────────────────────────────────────

    def run(self):
        """Execute the full simulation."""
        print(f"\n🚀 Starting simulation — {self.n_cycles} cycles × {len(self.robots)} robots")
        print("─" * 70)

        all_robots = dict(self.robots)  # snapshot; children added during run

        for cycle in range(1, self.n_cycles + 1):
            cycle_log = {"cycle": cycle, "events": []}

            for robot_name, robot in list(all_robots.items()):
                # Perception → Action
                action_id, action_name, sentiment = robot.perceive_and_act(seed=cycle * 37 + robot.id)

                if self.verbose:
                    mw = robot.action_log[-1]["modal_weights"]
                    mw_str = " ".join(f"{m:.2f}" for m in mw)
                    res = robot.resources
                    print(
                        f"  Cycle {cycle:03d} | {robot.domain_cfg['emoji']} {robot_name:<28} | "
                        f"{action_name:<28} | mood={sentiment.mood:<8} | "
                        f"mat={res.raw_materials:5.1f} parts={res.component_parts:3d} "
                        f"nrg={res.energy:4.1f} | modal_w=[{mw_str}]"
                    )

                cycle_log["events"].append({
                    "robot": robot_name,
                    "action": action_name,
                    "sentiment": sentiment.format(),
                })

                # Replication check
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

            # Brief pause for readability in live output
            if self.verbose:
                time.sleep(0.005)

        self._print_final_report(all_robots)
        return all_robots

    # ── Reporting ─────────────────────────────────────────────────────────

    def _print_final_report(self, all_robots: Dict):
        print("\n" + "═" * 70)
        print("  📊  FINAL UNIVERSE REPORT")
        print("═" * 70)

        total_parts = 0
        total_replications = 0
        domain_factories: Dict[str, int] = defaultdict(int)

        for name, robot in all_robots.items():
            s = robot.summary()
            total_parts += s["final_resources"]["parts"]
            total_replications += s["children"]
            for action, count in s["action_counts"].items():
                domain_factories[f"{robot.domain}:{action}"] += count

            emoji = DOMAINS[robot.domain]["emoji"]
            tier = "👶" if robot.is_child else "🤖"
            print(f"\n  {tier} {emoji} {name}")
            print(f"     Domain    : {robot.domain}")
            if robot.parent_name:
                print(f"     Parent    : {robot.parent_name}")
            print(f"     Cycles    : {s['cycles']}")
            print(f"     Children  : {s['children']}")
            print(f"     Actions   : {s['action_counts']}")
            print(f"     Mood      : {s['mood_distribution']}")
            print(f"     Resources : {s['final_resources']}")

        print(f"\n{'─' * 70}")
        print(f"  🌐 Total robots active      : {len(all_robots)}")
        print(f"  🧬 Total replications       : {total_replications}")
        print(f"  ⚙️  Total component parts    : {total_parts}")
        print(f"  🌌 Simulation complete.")
        print("═" * 70)

    def export_log(self, path: str):
        """Save the full hive log to JSON."""
        with open(path, "w") as f:
            json.dump(self.hive_log, f, indent=2)
        print(f"\n📝 Hive log saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
#  UTILITIES — STANDALONE INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def load_production_stack(model_dir: str) -> Tuple[PerceptionPipeline, Dict]:
    """
    Convenience function: load all models and return a ready-to-use
    PerceptionPipeline for embedding arbitrary sensor data.

    Usage:
        pipeline, info = load_production_stack("/path/to/models")
        bundle  = pipeline.make_synthetic_bundle(available_modalities=["text","audio"])
        emb, weights, details = pipeline.perceive(bundle)
    """
    norm = FeatureNormalizer(os.path.join(model_dir, NORM_JSON))
    holosyn = HoloSynHeads()
    student_hf = StudentDistilledHeadsHF()
    student_basic = StudentDistilledHeadsBasic()

    for cls, path in [
        (holosyn, os.path.join(model_dir, HOLOSYN_TORCHSCRIPT)),
        (student_hf, os.path.join(model_dir, STUDENT_HF_TS)),
        (student_basic, os.path.join(model_dir, STUDENT_BASIC_TS)),
    ]:
        cls.load_torchscript(path)

    pipeline = PerceptionPipeline(norm, holosyn, student_hf, student_basic)
    info = {
        "text_emb_dim": HoloSynHeads.TEXT_DIM,
        "img_emb_dim": HoloSynHeads.IMG_DIM,
        "aud_emb_dim": HoloSynHeads.AUD_DIM,
        "vid_emb_dim": HoloSynHeads.VID_DIM,
        "fused_dim": 256,
        "norm_features": 789,
        "modal_scores_dim": 4,
    }
    return pipeline, info


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Universal Constructor Robot Framework")
    parser.add_argument("--model-dir", default=MODEL_DIR,
                        help="Directory containing .pt and .json model files")
    parser.add_argument("--cycles", type=int, default=60,
                        help="Number of simulation cycles (default: 60)")
    parser.add_argument("--domains", nargs="+", choices=list(DOMAINS.keys()),
                        help="Domains to activate (default: all)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-cycle output")
    parser.add_argument("--export", default=None,
                        help="Path to export hive log JSON")
    args = parser.parse_args()

    uc = UniversalConstructor(
        model_dir=args.model_dir,
        domains=args.domains,
        n_cycles=args.cycles,
        verbose=not args.quiet,
    )
    uc.run()

    if args.export:
        uc.export_log(args.export)