"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PLANET FACTORY: PHASE 5 - MULTI-AGENT ECOSYSTEM               ║
║                                                                              ║
║  A practical, agentic planet-building system. Integrates Heterogeneous       ║
║  entities: AI Machines (Extractors), AI Robots (Constructors), Analog        ║
║  Sensors, and Human Operators (RLHF). Features a shared physical economy.    ║
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
            def info(self, msg): print(f"[{msg}")  # Cleaned up for custom ecosystem formatting

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
#  1. THE MODULAR CONTINUAL LEARNING SNN (Phase 4 Upgraded)
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

        # Namespace fixes: mecherror and panicsignal (removed underscores to avoid Brian2 run namespace conflicts)
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

        self.S.taupre = 20 * b2.ms;
        self.S.taupost = 20 * b2.ms
        self.S.tau_c = 50 * b2.ms;
        self.S.tau_d = 150 * b2.ms
        self.S.base_lr = 1.5 * b2.Hz

        if self.modules.get("homeostasis"): self.S.w_decay = 0.02 * b2.Hz
        if self.modules.get("heterosynaptic"):
            self.S.hetero_decay = 0.05 * b2.Hz
            self.S.global_activity = 0.0
        if self.modules.get("predictive_coding"):
            self.S.mech_lr = 3.0 * b2.Hz
            self.S.mecherror = 0.0
        if self.modules.get("noradrenaline"):
            self.S.panic_lr = 15.0 * b2.Hz
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
#  2. PLANET ENVIRONMENT & HUMAN OPERATOR
# ═════════════════════════════════════════════════════════════════════════════

class PlanetEnvironment:
    """Analog sensor array and global economic ledger."""

    def __init__(self):
        self.raw_materials = 0.0
        self.planet_infrastructure = 0.0
        self.global_energy = 1000.0
        self.atmospheric_temp = 25.0

    def print_state(self):
        print(
            f"🌍 PLANET STATE | Infra: {self.planet_infrastructure:.1f} | Materials: {self.raw_materials:.1f} | Energy: {self.global_energy:.1f} | Temp: {self.atmospheric_temp:.1f}C")


class HumanOperator:
    """Provides asynchronous Human Feedback (RLHF) and Overrides."""

    def assess_ecosystem(self, env: PlanetEnvironment) -> dict:
        commands = {"rlhf_reward": 0.0, "override_action": None}

        # Human praises the robots if infrastructure is growing rapidly
        if env.planet_infrastructure > 0 and np.random.rand() > 0.8:
            print("👨‍🔧 HUMAN FEEDBACK: 'Great progress!' (+2.0 Massive Dopamine Boost)")
            commands["rlhf_reward"] = 2.0

        # Human panic overrides if energy is critically low
        if env.global_energy < 850.0 and np.random.rand() > 0.7:
            print("👨‍🔧 HUMAN OVERRIDE: 'Energy low! Everyone switch to Recharge mode!'")
            commands["override_action"] = 2  # Force rest/recharge

        return commands


# ═════════════════════════════════════════════════════════════════════════════
#  3. AGENTIC CYBER-NODE (Heterogeneous Ecosystem Member)
# ═════════════════════════════════════════════════════════════════════════════

class AgenticCyberNode(Node):
    def __init__(self, name, role, env_ref: PlanetEnvironment, human_ref: HumanOperator):
        super().__init__(name)
        self.role = role  # "Extractor" or "Constructor"
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

        self.emoji = "⛏️ " if role == "Extractor" else "🏗️ "

    def cognitive_loop(self):
        """Agentic perception-action-learning tick."""
        # 1. Perception
        bundle = ModalityBundle(
            E_t_aug=torch.randn(1, 528), E_i=torch.zeros(1, 2048),
            E_a=torch.randn(1, 2048), E_v=torch.randn(1, 2048),
            H=torch.zeros(1, 32, 3), feat_hf=torch.randn(1, 789),
            feat_basic=torch.randn(1, 5), mask=torch.ones(1, 4),
        )
        fused_emb, _, _ = self.perception.perceive(bundle)

        # 2. Predictive Coding (Mechanical Error)
        actual_temp_delta = self.current_temp - self.previous_temp
        expected_delta = 5.0 if self.last_action_taken in [0, 1] else -5.0
        if self.last_action_taken == -1: expected_delta = -15.0

        mech_error_signal = 0.0
        if self.last_action_taken != -1:
            mech_error_signal = np.clip((actual_temp_delta - expected_delta) / 10.0, -1.0, 1.0)

            # 3. Human-In-The-Loop (RLHF)
        human_cmds = self.human.assess_ecosystem(self.env)
        if human_cmds["rlhf_reward"] > 0: self.last_reward += human_cmds["rlhf_reward"]

        # 4. SNN Inference
        proposed_action = self.head.predict_and_learn(
            z=fused_emb, reward_signal=self.last_reward, attention_signal=self.last_attention,
            mood_signal=self.last_mood, mech_error_signal=mech_error_signal, panic_signal=self.last_panic
        )

        if human_cmds["override_action"] is not None:
            proposed_action = human_cmds["override_action"]

        # 5. Amygdala Safety Check
        is_safe, reason = self.governor.validate_action(proposed_action, {"temperature": self.current_temp})
        self.previous_temp = self.current_temp

        # 6. Actuation & Ecosystem Economics
        self.last_attention = 1.0 + min(2.0, abs(actual_temp_delta) * 0.5)

        if is_safe:
            self.last_mood = -0.1
            self.last_panic = 0.0

            # Base Reward + Relief Bonus
            self.last_reward = 1.0 if (self.current_temp / 85.0) < 0.75 else 0.2
            if actual_temp_delta < 0: self.last_reward += 0.5

            self.last_action_taken = proposed_action

            # ── AGENTIC ROLE EXECUTION ──
            if proposed_action == 0:
                action_name = "Heavy Work"
                self.current_temp += (7.0 + self.machine_wear)
                self.machine_wear += 0.2
                self.env.global_energy -= 5.0
                if self.role == "Extractor": self.env.raw_materials += 10.0

            elif proposed_action == 1:
                action_name = "Precision Work"
                self.current_temp += (4.0 + self.machine_wear)
                self.machine_wear += 0.1
                self.env.global_energy -= 2.0
                if self.role == "Constructor" and self.env.raw_materials >= 5.0:
                    self.env.raw_materials -= 5.0
                    self.env.planet_infrastructure += 1.0
                    self.last_reward += 1.0  # Intrinsic reward for successfully building
                elif self.role == "Constructor":
                    action_name = "Idle (No Materials)"

            elif proposed_action == 2:
                action_name = "Recharge/Maintain"
                self.current_temp = max(40.0, self.current_temp - 8.0)
                self.machine_wear = max(0.0, self.machine_wear - 0.2)
                self.env.global_energy += 1.0

            self.get_logger().info(
                f"{self.emoji}{self.name}] {action_name} | Temp: {self.current_temp:.1f}C | Dopamine: {self.last_reward:.1f}")
        else:
            self.last_mood = +0.5
            self.last_reward = 0.0
            self.last_panic = 1.0  # Targeted Noradrenaline
            self.last_action_taken = -1

            self.get_logger().warn(f"{self.emoji}{self.name}] BLOCKED ({reason})! Noradrenaline injected.")
            self.current_temp = max(40.0, self.current_temp - 15.0)
            self.machine_wear *= 0.8

        # ═════════════════════════════════════════════════════════════════════════════


#  ENTRY POINT: PLANET SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

def run_planet_simulation():
    print("\n" + "═" * 70)
    print(" 🌌 PLANET FACTORY: MULTI-AGENT CYBER-PHYSICAL ECOSYSTEM")
    print("═" * 70)

    b2.start_scope()

    # Instantiate the Planet and Human
    planet = PlanetEnvironment()
    human = HumanOperator()

    # Instantiate Diverse Agents
    agents = [
        AgenticCyberNode("Titan-Drill", "Extractor", planet, human),
        AgenticCyberNode("Aero-Swarm-1", "Constructor", planet, human)
    ]

    print("\n🚀 COMMENCING PLANET COLONIZATION...")
    for tick in range(1, 31):
        print(f"\n--- Cycle {tick} ---")
        for agent in agents:
            agent.cognitive_loop()

        # Environmental Update
        if tick % 5 == 0:
            planet.print_state()

        time.sleep(0.05)

    print("\n🏁 ECOSYSTEM SIMULATION COMPLETE.")
    planet.print_state()
    print("═" * 70)


if __name__ == '__main__':
    if HAS_ROS2:
        print("Please run offline simulation for ecosystem test.")
    else:
        run_planet_simulation()