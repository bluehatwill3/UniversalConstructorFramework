"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PLANET FACTORY: PHASE 5 - MULTI-AGENT ECOSYSTEM               ║
║                                                                              ║
║  A practical, agentic planet-building system. Integrates Heterogeneous       ║
║  entities across Self-Driving, Logistics, Production, and Mining. Features   ║
║  a shared physical economy, Human Operators (RLHF), and Modular SNNs.        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import warnings

# Try to import OpenCV for image processing
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    warnings.warn("OpenCV (cv2) not installed. Image processing will be bypassed.")

# Import the core UCF architecture
from ucf import (
    PerceptionPipeline, FeatureNormalizer, HoloSynHeads,
    StudentDistilledHeadsHF, StudentDistilledHeadsBasic, ModalityBundle,
    SafetyGovernor, IndustrialThresholds
)

# ── 🧠 NEUROMORPHIC BACKEND ──────────────────────────────────────────────────
try:
    import brian2 as b2

    b2.prefs.codegen.target = 'numpy'
    HAS_NEUROMORPHIC = True
except ImportError:
    HAS_NEUROMORPHIC = False

# ── 🌐 ROS 2 NERVOUS SYSTEM ──────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, LaserScan, Imu, JointState
    from std_msgs.msg import Float32, String, Float32MultiArray
    from geometry_msgs.msg import Twist, WrenchStamped
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from cv_bridge import CvBridge

    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False


    # Mock ROS 2 Classes
    class Node:
        def __init__(self, name): self.name = name

        class MockLogger:
            def info(self, msg): print(f"[{msg}")

            def warn(self, msg): print(f"[\033[93mWARN\033[0m] {msg}")

        def get_logger(self): return self.MockLogger()

        def create_subscription(self, *args, **kwargs): pass

        def create_publisher(self, *args, **kwargs):
            class MockPub:
                def publish(self, msg): pass

            return MockPub()

        def create_timer(self, *args, **kwargs): pass


    class Twist:
        class Vector3: x = 0.0; y = 0.0; z = 0.0

        def __init__(self): self.linear = self.Vector3(); self.angular = self.Vector3()


    class Float32:
        pass


    class String:
        pass


    class Image:
        pass


    class LaserScan:
        pass


    class Imu:
        pass


    class JointState:
        pass


    class Float32MultiArray:
        pass


    class WrenchStamped:
        pass


    class JointTrajectory:
        pass


    class JointTrajectoryPoint:
        pass


    class CvBridge:
        pass


# ═════════════════════════════════════════════════════════════════════════════
#  1. THE MODULAR CONTINUAL LEARNING SNN (Optimized for Multi-Domain)
# ═════════════════════════════════════════════════════════════════════════════

class AdvancedNeuromorphicHead(nn.Module):
    def __init__(self, n_actions: int, learning_modules: dict = None):
        super().__init__()
        self.n_actions = n_actions
        self.modules = learning_modules or {
            "stdp": True, "reward": True, "attention": True,
            "homeostasis": True, "heterosynaptic": True,
            "predictive_coding": True, "noradrenaline": True
        }

        if not HAS_NEUROMORPHIC: return

        self.P = b2.PoissonGroup(256, rates=np.zeros(256) * b2.Hz)

        eqs_neurons = '''
        dv/dt = (I - v) / (10*ms) : 1
        I : 1
        mood_modifier : 1  
        '''
        self.G = b2.NeuronGroup(n_actions, eqs_neurons, threshold='v > (1.0 + mood_modifier)', reset='v=0',
                                refractory=2 * b2.ms, method='euler')

        base_eqs = '''
        dApre/dt = -Apre / taupre : 1 (event-driven)
        dApost/dt = -Apost / taupost : 1 (event-driven)
        dc/dt = -c / tau_c : 1 (clock-driven)  
        dx/dt = (1 - x) / tau_d : 1 (clock-driven)
        '''

        dw_components = []
        if self.modules.get("stdp"):
            core = "c"
            if self.modules.get("attention"):
                core = "(base_lr * attention) * " + core
            else:
                core = "base_lr * " + core
            if self.modules.get("reward"): core += " * reward"
            dw_components.append(core)

        if self.modules.get("homeostasis"): dw_components.append("- w_decay * w")
        if self.modules.get("heterosynaptic"): dw_components.append("- hetero_decay * global_activity * w")
        if self.modules.get("predictive_coding"): dw_components.append("- mech_lr * mecherror * c")
        if self.modules.get("noradrenaline"): dw_components.append("- panic_lr * panicsignal * c")

        dw_dt_eq = "dw/dt = " + " ".join(
            dw_components) + " : 1 (clock-driven)" if dw_components else "dw/dt = 0 : 1 (clock-driven)"

        shared_vars = '''
        taupre : second (shared)
        taupost : second (shared)
        tau_c : second (shared)
        tau_d : second (shared)
        base_lr : 1/second (shared)
        '''
        if self.modules.get("reward"): shared_vars += "reward : 1 (shared)\n"
        if self.modules.get("attention"): shared_vars += "attention : 1 (shared)\n"
        if self.modules.get("homeostasis"): shared_vars += "w_decay : 1/second (shared)\n"
        if self.modules.get("heterosynaptic"):
            shared_vars += "hetero_decay : 1/second (shared)\n"
            shared_vars += "global_activity : 1 (shared)\n"
        if self.modules.get("predictive_coding"):
            shared_vars += "mecherror : 1 (shared)\n"
            shared_vars += "mech_lr : 1/second (shared)\n"
        if self.modules.get("noradrenaline"):
            shared_vars += "panicsignal : 1 (shared)\n"
            shared_vars += "panic_lr : 1/second (shared)\n"

        syn_eqs = base_eqs + dw_dt_eq + shared_vars

        self.S = b2.Synapses(self.P, self.G, model=syn_eqs,
                             on_pre='''v_post += w * x
                                       x *= 0.8  
                                       Apre += 0.01
                                       c += Apost''',
                             on_post='''Apost -= 0.01
                                        c += Apre''',
                             method='euler')

        self.S.connect()
        self.S.w = 'rand() * 0.3'
        self.S.x = 1.0

        # Fine-tuned for multi-domain responsiveness
        self.S.taupre = 20 * b2.ms;
        self.S.taupost = 20 * b2.ms
        self.S.tau_c = 50 * b2.ms;
        self.S.tau_d = 120 * b2.ms
        self.S.base_lr = 1.8 * b2.Hz

        if self.modules.get("homeostasis"): self.S.w_decay = 0.02 * b2.Hz
        if self.modules.get("heterosynaptic"):
            self.S.hetero_decay = 0.05 * b2.Hz
            self.S.global_activity = 0.0
        if self.modules.get("predictive_coding"):
            self.S.mech_lr = 3.5 * b2.Hz  # Aggressive correction for physical safety
            self.S.mecherror = 0.0
        if self.modules.get("noradrenaline"):
            self.S.panic_lr = 20.0 * b2.Hz  # Intense, targeted unlearning on failure
            self.S.panicsignal = 0.0
        if self.modules.get("reward"): self.S.reward = 0.0
        if self.modules.get("attention"): self.S.attention = 1.0

        self.G.mood_modifier = 0.0
        self.M = b2.SpikeMonitor(self.G)
        self.net = b2.Network(self.P, self.G, self.S, self.M)
        self.last_counts = np.zeros(n_actions)

    def predict_and_learn(self, z, reward_signal, attention_signal, mood_signal, mech_error_signal, panic_signal=0.0):
        if not HAS_NEUROMORPHIC: return 0
        self.P.rates = np.clip(z.squeeze().cpu().numpy(), 0, 1) * 100 * b2.Hz

        if self.modules.get("reward"): self.S.reward = reward_signal
        if self.modules.get("attention"): self.S.attention = attention_signal
        if self.modules.get("predictive_coding"): self.S.mecherror = mech_error_signal
        if self.modules.get("noradrenaline"): self.S.panicsignal = panic_signal
        if self.modules.get("heterosynaptic"): self.S.global_activity = float(
            np.sum(self.last_counts) / max(1, self.n_actions))
        self.G.mood_modifier = mood_signal

        self.net.run(20 * b2.ms)

        current_counts = np.array(self.M.count)
        step_counts = current_counts - self.last_counts
        self.last_counts = current_counts.copy()

        if np.sum(step_counts) > 0: return int(np.argmax(step_counts))
        return np.random.randint(0, self.n_actions)


# ═════════════════════════════════════════════════════════════════════════════
#  2. PLANET ENVIRONMENT, ECONOMICS, & HUMAN OPERATOR
# ═════════════════════════════════════════════════════════════════════════════

class PlanetEnvironment:
    """Analog sensor array and global economic ledger tracking multi-industry metrics."""

    def __init__(self):
        self.raw_materials = 50.0
        self.manufactured_goods = 0.0
        self.cargo_delivered = 0.0
        self.miles_driven = 0.0
        self.planet_infrastructure = 0.0
        self.global_energy = 2000.0

    def print_state(self):
        print("═" * 70)
        print(
            f"🌍 PLANET STATE | Energy: {self.global_energy:.1f} | Materials: {self.raw_materials:.1f} | Goods: {self.manufactured_goods:.1f}")
        print(f"   LOGISTICS    | Miles Driven: {self.miles_driven:.1f} | Cargo Delivered: {self.cargo_delivered:.1f}")
        print("═" * 70)


class HumanOperator:
    """Provides asynchronous Human Feedback (RLHF) and Overrides."""

    def assess_ecosystem(self, env: PlanetEnvironment) -> dict:
        commands = {"rlhf_reward": 0.0, "override_action": None}

        # Human praises if manufacturing or logistics is flowing well
        if (env.manufactured_goods > 10 or env.cargo_delivered > 5) and np.random.rand() > 0.85:
            print("👨‍🔧 HUMAN FEEDBACK: 'Excellent throughput!' (+2.0 Massive Dopamine Boost)")
            commands["rlhf_reward"] = 2.0

        # Human panic overrides if energy is critically low
        if env.global_energy < 1500.0 and np.random.rand() > 0.9:
            print("👨‍🔧 HUMAN OVERRIDE: 'Energy low! All units, switch to Yield/Recharge!'")
            commands["override_action"] = 2  # Action 2 is generally 'Rest/Yield' across all domains

        return commands


# ═════════════════════════════════════════════════════════════════════════════
#  3. DOMAIN CONFIGURATIONS (FSD, Logistics, Production, Extraction)
# ═════════════════════════════════════════════════════════════════════════════

def effect_extract(env, amt): env.raw_materials += amt


def effect_produce(env, cost):
    if env.raw_materials >= cost:
        env.raw_materials -= cost
        env.manufactured_goods += 1.0
        return True
    return False


def effect_logistics(env, amt):
    if env.manufactured_goods >= amt:
        env.manufactured_goods -= amt
        env.cargo_delivered += amt
        return True
    return False


def effect_fsd(env, miles): env.miles_driven += miles


DOMAIN_CONFIGS = {
    "Self-Driving": {
        "emoji": "🚗",
        0: {"name": "Accelerate", "heat": 6.0, "wear": 0.1, "energy": 4.0, "effect": lambda e: effect_fsd(e, 5.0)},
        1: {"name": "Navigate Traffic", "heat": 4.0, "wear": 0.05, "energy": 2.0,
            "effect": lambda e: effect_fsd(e, 2.0)},
        2: {"name": "Brake/Yield", "heat": -6.0, "wear": -0.1, "energy": -1.0, "effect": lambda e: None}
    },
    "Logistics": {
        "emoji": "🚁",
        0: {"name": "Load & Lift", "heat": 8.0, "wear": 0.2, "energy": 6.0,
            "effect": lambda e: effect_logistics(e, 1.0)},
        1: {"name": "Transport Cargo", "heat": 5.0, "wear": 0.1, "energy": 3.0,
            "effect": lambda e: effect_fsd(e, 10.0)},
        2: {"name": "Dock/Recharge", "heat": -8.0, "wear": -0.2, "energy": -5.0, "effect": lambda e: None}
    },
    "Production": {
        "emoji": "🏭",
        0: {"name": "Mill/Forge", "heat": 10.0, "wear": 0.3, "energy": 8.0, "effect": lambda e: effect_produce(e, 5.0)},
        1: {"name": "Precision Assembly", "heat": 3.0, "wear": 0.05, "energy": 2.0,
            "effect": lambda e: effect_produce(e, 2.0)},
        2: {"name": "Cool Down Cycle", "heat": -12.0, "wear": -0.1, "energy": -2.0, "effect": lambda e: None}
    },
    "Extractor": {
        "emoji": "⛏️",
        0: {"name": "Heavy Drill", "heat": 9.0, "wear": 0.25, "energy": 7.0,
            "effect": lambda e: effect_extract(e, 15.0)},
        1: {"name": "Surface Scan", "heat": 2.0, "wear": 0.02, "energy": 1.0,
            "effect": lambda e: effect_extract(e, 2.0)},
        2: {"name": "System Vent", "heat": -10.0, "wear": -0.1, "energy": -2.0, "effect": lambda e: None}
    }
}


# ═════════════════════════════════════════════════════════════════════════════
#  4. AGENTIC CYBER-NODE (Heterogeneous Ecosystem Member)
# ═════════════════════════════════════════════════════════════════════════════

class AgenticCyberNode(Node):
    def __init__(self, name, role, env_ref: PlanetEnvironment, human_ref: HumanOperator):
        super().__init__(name.replace("-", "_").lower())
        self.node_name = name
        self.role = role
        self.domain = DOMAIN_CONFIGS[role]
        self.emoji = self.domain["emoji"]
        self.env = env_ref
        self.human = human_ref

        # Embodied State
        self.current_temp = 40.0
        self.previous_temp = 40.0
        self.machine_wear = 0.0

        self.last_reward = 0.0
        self.last_attention = 1.0
        self.last_mood = 0.0
        self.last_panic = 0.0
        self.last_action_taken = -1

        # Neural Architecture
        self.head = AdvancedNeuromorphicHead(n_actions=3)
        self.governor = SafetyGovernor(IndustrialThresholds(max_temp=85.0))
        self.perception = PerceptionPipeline(FeatureNormalizer(), HoloSynHeads(), StudentDistilledHeadsHF(),
                                             StudentDistilledHeadsBasic())

    def cognitive_loop(self):
        """Agentic perception-action-learning tick."""
        bundle = ModalityBundle(
            E_t_aug=torch.randn(1, 528), E_i=torch.zeros(1, 2048),
            E_a=torch.randn(1, 2048), E_v=torch.randn(1, 2048),
            H=torch.zeros(1, 32, 3), feat_hf=torch.randn(1, 789),
            feat_basic=torch.randn(1, 5), mask=torch.ones(1, 4),
        )
        fused_emb, _, _ = self.perception.perceive(bundle)

        # ── 1. PREDICTIVE CODING: Dynamic Expectation based on Role ──
        actual_temp_delta = self.current_temp - self.previous_temp
        expected_delta = 0.0

        if self.last_action_taken in [0, 1, 2]:
            expected_delta = self.domain[self.last_action_taken]["heat"]
        elif self.last_action_taken == -1:
            expected_delta = -15.0  # Expected cooling from governor block

        mech_error_signal = 0.0
        if self.last_action_taken != -1:
            mech_error_signal = np.clip((actual_temp_delta - expected_delta) / 10.0, -1.0, 1.0)

            # ── 2. HUMAN-IN-THE-LOOP & SNN INFERENCE ──
        human_cmds = self.human.assess_ecosystem(self.env)
        if human_cmds["rlhf_reward"] > 0: self.last_reward += human_cmds["rlhf_reward"]

        proposed_action = self.head.predict_and_learn(
            z=fused_emb, reward_signal=self.last_reward, attention_signal=self.last_attention,
            mood_signal=self.last_mood, mech_error_signal=mech_error_signal, panic_signal=self.last_panic
        )

        if human_cmds["override_action"] is not None:
            proposed_action = human_cmds["override_action"]

        # ── 3. AMYGDALA SAFETY CHECK ──
        is_safe, reason = self.governor.validate_action(proposed_action, {"temperature": self.current_temp})
        self.previous_temp = self.current_temp

        # ── 4. ACTUATION & ECONOMICS ──
        self.last_attention = 1.0 + min(2.0, abs(actual_temp_delta) * 0.5)

        if is_safe:
            self.last_mood = -0.1
            self.last_panic = 0.0

            # Base Reward + Relief Bonus
            self.last_reward = 1.0 if (self.current_temp / 85.0) < 0.75 else 0.2
            if actual_temp_delta < 0: self.last_reward += 0.5

            self.last_action_taken = proposed_action
            action_cfg = self.domain[proposed_action]

            # Apply Physical Consequences
            self.current_temp += (action_cfg["heat"] + self.machine_wear)
            self.machine_wear = max(0.0, self.machine_wear + action_cfg["wear"])
            self.env.global_energy -= action_cfg["energy"]

            # Apply Environmental Economy Consequences
            success = action_cfg["effect"](self.env)

            action_status = action_cfg["name"]
            if success is False:
                action_status += " (Waiting on Supply Chain)"
                self.last_reward -= 0.5  # Small penalty for supply chain stall
            elif success is True:
                self.last_reward += 0.5  # Intrinsic reward for productive output

            self.get_logger().info(
                f"{self.emoji} {self.node_name:<16} | {action_status:<28} | Temp: {self.current_temp:>4.1f}C | Wear: {self.machine_wear:>3.1f} | Dopa: {self.last_reward:>4.1f}")
        else:
            self.last_mood = +0.5
            self.last_reward = 0.0
            self.last_panic = 1.0  # Targeted Noradrenaline (Burn the synapses that caused overheating)
            self.last_action_taken = -1

            self.get_logger().warn(f"{self.emoji} {self.node_name:<16} | BLOCKED ({reason})! Noradrenaline injected.")
            self.current_temp = max(40.0, self.current_temp - 15.0)
            self.machine_wear *= 0.8  # Simulated maintenance during downtime


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT: PLANET SIMULATION TEST (FSD, Logistics, Production)
# ═════════════════════════════════════════════════════════════════════════════

def run_planet_simulation():
    print("\n" + "═" * 80)
    print(" 🌌 PLANET FACTORY: MULTI-INDUSTRY CYBER-PHYSICAL ECOSYSTEM TEST")
    print("═" * 80)

    b2.start_scope()

    planet = PlanetEnvironment()
    human = HumanOperator()

    # Instantiate Diverse Multi-Industry Agents
    agents = [
        AgenticCyberNode("Titan-Drill", "Extractor", planet, human),
        AgenticCyberNode("CNC-Forge-Alpha", "Production", planet, human),
        AgenticCyberNode("Aero-Swarm-1", "Logistics", planet, human),
        AgenticCyberNode("Auto-Hauler-X", "Self-Driving", planet, human)
    ]

    print("\n🚀 COMMENCING PLANET COLONIZATION...")
    for tick in range(1, 26):
        print(f"\n--- Cycle {tick} ---")
        for agent in agents:
            agent.cognitive_loop()

        # Environmental Update
        if tick % 5 == 0:
            planet.print_state()

        time.sleep(0.05)

    print("\n🏁 ECOSYSTEM SIMULATION COMPLETE.")
    planet.print_state()
    print("═" * 80)


if __name__ == '__main__':
    if HAS_ROS2:
        print("Please run offline simulation for ecosystem test.")
    else:
        run_planet_simulation()