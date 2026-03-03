import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from collections import defaultdict
from datetime import datetime

# ── BRAIN2 CONFIGURATION ─────────────────────────────────────────────────────
try:
    import brian2 as b2

    b2.prefs.codegen.target = 'numpy'  # Bypass C++ compiler issues
    HAS_NEUROMORPHIC = True
except ImportError:
    HAS_NEUROMORPHIC = False


# ═════════════════════════════════════════════════════════════════════════════
# 1. SAFETY & MODALITY DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ModalityBundle:
    """Consolidated sensor packet for industrial perception."""
    E_t_aug: torch.Tensor
    E_i: Optional[torch.Tensor]
    E_a: torch.Tensor
    E_v: torch.Tensor
    H: torch.Tensor
    feat_hf: torch.Tensor
    mask: torch.Tensor
    meta: Dict = field(default_factory=dict)


@dataclass
class IndustrialThresholds:
    """Physical limits for hardware protection."""
    max_temp: float = 85.0
    vibration_limit: float = 0.8
    min_energy: float = 10.0


class SafetyGovernor:
    """The safety interceptor for industrial actions."""

    def __init__(self, thresholds: IndustrialThresholds):
        self.limits = thresholds

    def validate_action(self, action_id: int, sensors: Dict) -> Tuple[bool, str]:
        """Returns (is_safe, reason) based on real-time metrics."""
        if sensors.get("temperature", 0) > self.limits.max_temp:
            return False, "THERMAL_OVERLOAD"
        if sensors.get("vibration", 0) > self.limits.vibration_limit:
            return False, "MECHANICAL_STRESS"
        if sensors.get("energy", 100) < self.limits.min_energy:
            return False, "LOW_POWER"
        return True, "SAFE"


# ═════════════════════════════════════════════════════════════════════════════
# 2. NEUROMORPHIC POLICY HEAD
# ═════════════════════════════════════════════════════════════════════════════

class NeuromorphicActionHead(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def predict(self, z: torch.Tensor) -> int:
        """Inference utilizing Spiking Neural Network logic if available."""
        if HAS_NEUROMORPHIC:
            b2.start_scope()
            snn_input_rates = np.clip(z.squeeze().cpu().numpy(), 0, 1) * 100 * b2.Hz
            P = b2.PoissonGroup(256, rates=snn_input_rates)  # Fixed namespace
            G = b2.NeuronGroup(self.n_actions, 'dv/dt = (I - v) / (10*ms) : 1\nI : 1',
                               threshold='v>1', reset='v=0', method='exact')
            S = b2.Synapses(P, G, 'w : 1', on_pre='v_post += w')
            S.connect()
            S.w = 'rand() * 0.1'
            M = b2.SpikeMonitor(G)
            b2.run(20 * b2.ms)
            return int(np.argmax(M.count)) if np.sum(M.count) > 0 else 0
        return int(self.net(z).argmax(dim=-1).item())


# ═════════════════════════════════════════════════════════════════════════════
# 3. INDUSTRIAL ROBOT ENTITY
# ═════════════════════════════════════════════════════════════════════════════

class ResourceLedger:
    def __init__(self):
        self.energy = 100.0
        self.parts_produced = 0


class IndustrialRobot:
    def __init__(self, name: str, domain: str, n_actions: int):
        self.name = name
        self.domain = domain
        self.action_head = NeuromorphicActionHead(n_actions)
        self.resources = ResourceLedger()
        self.action_log = []

    def apply_mutation(self, power: float = 0.05):
        """Applies Gaussian noise to simulate genetic drift."""
        with torch.no_grad():
            for param in self.action_head.parameters():
                param.add_(torch.randn_like(param) * power)


# ═════════════════════════════════════════════════════════════════════════════
# 4. SIMULATION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class UniversalConstructor:
    def __init__(self, cycles: int = 50):
        self.cycles = cycles
        self.governor = SafetyGovernor(IndustrialThresholds())
        self.robots = [IndustrialRobot("Alpha-Forge", "production", 3)]

    def run(self):
        print(f"🚀 Initializing Hive Control for {self.cycles} cycles...")
        for c in range(1, self.cycles + 1):
            for robot in self.robots:
                # 1. Perception
                state = torch.randn(1, 256)  # Fused embedding proxy
                proposed_id = robot.action_head.predict(state)

                # 2. Real-world sensor metrics proxy
                metrics = {
                    "temperature": 75.0 + (np.random.rand() * 15),
                    "vibration": np.random.rand() * 0.9,
                    "energy": robot.resources.energy
                }

                # 3. Safety Check
                is_safe, reason = self.governor.validate_action(proposed_id, metrics)
                final_id = proposed_id if is_safe else 0

                # 4. Execution & Update
                robot.resources.energy -= 0.5 if is_safe else 0.1
                if is_safe: robot.resources.parts_produced += 1

                robot.action_log.append({"cycle": c, "safe": is_safe, "reason": reason})
                print(f" Cycle {c:02d} | {robot.name} | {'✅ SAFE' if is_safe else f'🛑 BLOCKED ({reason})'}")


if __name__ == "__main__":
    system = UniversalConstructor(cycles=10)
    system.run()