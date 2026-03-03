from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any
from datetime import datetime
import torch
import numpy as np


# 1. We must define or import the Bundle so the Governor recognizes it
@dataclass
class ModalityBundle:
    E_t_aug: torch.Tensor
    E_i: Optional[torch.Tensor]
    E_a: torch.Tensor
    E_v: torch.Tensor
    H: torch.Tensor
    feat_hf: torch.Tensor
    feat_basic: torch.Tensor
    mask: torch.Tensor
    raw_image: Optional[torch.Tensor] = None
    meta: Dict = field(default_factory=dict)


@dataclass
class IndustrialThresholds:
    max_temp: float = 85.0
    max_torque: float = 450.0
    min_energy: float = 5.0
    vibration_limit: float = 0.8


class SafetyGovernor:
    def __init__(self, thresholds: IndustrialThresholds):
        self.limits = thresholds

    def validate_action(self, action_id: int, sensor_data: Dict) -> Tuple[bool, str]:
        temp = sensor_data.get("temperature", 0.0)
        if temp > self.limits.max_temp:
            return False, f"OVERHEAT: {temp}°C"

        vibration = sensor_data.get("vibration", 0.0)
        if vibration > self.limits.vibration_limit:
            return False, f"VIBRATION_LIMIT: {vibration}"

        return True, "SAFE"


# 2. Now safe_perceive_and_act will recognize 'ModalityBundle'
def safe_perceive_and_act(robot: Any, governor: SafetyGovernor, bundle: ModalityBundle):
    # Logic for safety-checked execution goes here
    print(f"Checking safety for robot: {robot.name}")
    # ... (rest of the logic)
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import torch
import numpy as np

@dataclass
class IndustrialThresholds:
    """Safety limits for real-world industrial hardware."""
    max_temp: float = 85.0
    max_torque: float = 450.0
    min_energy: float = 5.0
    vibration_limit: float = 0.8


class SafetyGovernor:
    """Intercepts actions that violate physical safety constraints."""

    def __init__(self, thresholds: IndustrialThresholds):
        self.limits = thresholds

    def validate_action(self, action_id: int, sensor_data: Dict) -> Tuple[bool, str]:
        """Returns (Is_Safe, Reason)."""
        # Check Thermal Constraints (from aud_meta/temperature_sensor)
        temp = sensor_data.get("temperature", 0.0)
        if temp > self.limits.max_temp:
            return False, f"OVERHEAT_CRITICAL: {temp}°C"

        # Check Vibration/Stability (from haptic_seq variance)
        vibration = sensor_data.get("vibration", 0.0)
        if vibration > self.limits.vibration_limit:
            return False, f"UNSTABLE_VIBRATION: {vibration}G"

        # Check Energy levels
        energy = sensor_data.get("energy", 100.0)
        if energy < self.limits.min_energy:
            return False, "LOW_BATTERY_FORCED_STALL"

        return True, "SAFE"


# ── Updated Perceive and Act with Safety Injection ─────────────────────────

def safe_perceive_and_act(self, governor: SafetyGovernor, bundle: ModalityBundle):
    self.cycle += 1

    # 1. Perception
    fused_emb, _, _ = self.perception.perceive(bundle)

    # 2. Get Raw Action from Neuromorphic Head
    proposed_action_id = self.action_head.predict(fused_emb)

    # 3. Extract sensor context for the Governor
    current_sensors = {
        "temperature": bundle.meta.get("aud_meta", [0, 0, 0, 0])[2],  # Index 2 is temp
        "vibration": torch.std(bundle.H).item(),
        "energy": self.resources.energy
    }

    # 4. Filter Action
    is_safe, reason = governor.validate_action(proposed_action_id, current_sensors)

    if not is_safe:
        print(f"  🛑 SAFETY OVERRIDE [{self.name}]: Action {proposed_action_id} blocked. Reason: {reason}")
        final_action_id = 0  # Default to 'Maintain Params' or 'Conserve Energy'
    else:
        final_action_id = proposed_action_id

    # 5. Execute and Log
    action_name = self.domain_cfg["action_map"].get(final_action_id, "Unknown")
    self.resources.update_for_action(final_action_id, self.domain)
    return final_action_id, action_name