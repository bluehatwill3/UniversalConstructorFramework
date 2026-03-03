"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PLANET FACTORY: PHASE 7 - INTERPLANETARY STANDARDIZATION      ║
║                                                                              ║
║  Standardized for multi-asteroid deployment. Features ROS 2 Topic-based      ║
║  Stigmergy, Software Metamorphosis (to respect standard hardware chassis     ║
║  limitations), and multi-environment ambient entropy.                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import warnings
from collections import defaultdict

# Try to import OpenCV for image processing
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

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


# ── 🌐 ROS 2 NERVOUS SYSTEM & MOCK EVENT BUS ─────────────────────────────────
# To standardize offline testing for multi-planetary networking, we use a mock event bus
class MockEventBus:
    def __init__(self):
        self.topics = defaultdict(list)

    def subscribe(self, topic, callback):
        self.topics[topic].append(callback)

    def publish(self, topic, msg):
        for cb in self.topics[topic]: cb(msg)


mock_network = MockEventBus()

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


    # Standardized Mock ROS 2 Classes
    class Node:
        def __init__(self, name): self.name = name

        class MockLogger:
            def info(self, msg): print(f"[{msg}")

            def warn(self, msg): print(f"[\033[93mWARN\033[0m] {msg}")

            def error(self, msg): print(f"[\033[91mERROR\033[0m] {msg}")

        def get_logger(self): return self.MockLogger()

        def create_subscription(self, msg_type, topic, callback, qos):
            mock_network.subscribe(topic, callback)
            return topic

        def create_publisher(self, msg_type, topic, qos):
            class MockPub:
                def publish(self, msg): mock_network.publish(topic, msg)

            return MockPub()

        def create_timer(self, *args, **kwargs): pass


    class Twist:
        class Vector3: x = 0.0; y = 0.0; z = 0.0

        def __init__(self): self.linear = self.Vector3(); self.angular = self.Vector3()


    class Float32:
        def __init__(self, data=0.0): self.data = data


    class Float32MultiArray:
        def __init__(self, data=None): self.data = data or []


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


    class WrenchStamped:
        pass


    class JointTrajectory:
        pass


    class JointTrajectoryPoint:
        pass


    class CvBridge:
        pass


# ═════════════════════════════════════════════════════════════════════════════
#  1. THE MODULAR CONTINUAL LEARNING SNN
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
            self.S.mech_lr = 3.5 * b2.Hz
            self.S.mecherror = 0.0
        if self.modules.get("noradrenaline"):
            self.S.panic_lr = 20.0 * b2.Hz
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
#  2. PLANETARY BODY (Networked Environment Node)
# ═════════════════════════════════════════════════════════════════════════════

class PlanetaryBodyNode(Node):
    """Simulates a planet/asteroid environment publishing state to ROS topics."""

    def __init__(self, planet_id: str, ambient_temp: float, init_materials: float):
        super().__init__(f'planet_{planet_id}')
        self.planet_id = planet_id
        self.ambient_temp = ambient_temp

        self.raw_materials = init_materials
        self.planet_infrastructure = 0.0
        self.global_energy = 5000.0

        self.pheromones = {"danger": 0.0, "opportunity": 0.0}

        # Publishers
        self.pub_telemetry = self.create_publisher(Float32MultiArray, f'/{self.planet_id}/telemetry', 10)
        self.pub_stigmergy = self.create_publisher(Float32MultiArray, f'/{self.planet_id}/stigmergy', 10)

        # Subscribers for Swarm Interactions
        self.create_subscription(Float32MultiArray, f'/{self.planet_id}/agent_action', self.agent_action_callback, 10)

    def agent_action_callback(self, msg):
        """Processes economic/stigmergic impact from agents."""
        action_type, impact, danger_sp, opp_sp = msg.data
        if action_type == 1.0:  # Consume Energy
            self.global_energy -= impact
        elif action_type == 2.0:  # Mine Materials
            self.raw_materials += impact
        elif action_type == 3.0:  # Build Infra
            if self.raw_materials >= impact:
                self.raw_materials -= impact
                self.planet_infrastructure += 1.0

        # Pheromone Spikes
        self.pheromones["danger"] += danger_sp
        self.pheromones["opportunity"] += opp_sp

    def environment_tick(self):
        # Environmental Decay
        self.pheromones["danger"] = max(0.0, self.pheromones["danger"] - 1.5)
        self.pheromones["opportunity"] = max(0.0, self.pheromones["opportunity"] - 0.5)

        # Broadcast State
        tel_msg = Float32MultiArray(
            data=[self.global_energy, self.raw_materials, self.planet_infrastructure, self.ambient_temp])
        self.pub_telemetry.publish(tel_msg)

        stig_msg = Float32MultiArray(data=[self.pheromones["danger"], self.pheromones["opportunity"]])
        self.pub_stigmergy.publish(stig_msg)

    def print_state(self):
        print(
            f"🌍 [{self.planet_id.upper()}] Energy: {self.global_energy:.0f} | Mats: {self.raw_materials:.0f} | Infra: {self.planet_infrastructure:.0f} | Danger: {self.pheromones['danger']:.1f}")


# ═════════════════════════════════════════════════════════════════════════════
#  3. STANDARDIZED CHASSIS & SOFTWARE METAMORPHOSIS MODES
# ═════════════════════════════════════════════════════════════════════════════

CHASSIS_CONFIGS = {
    "Hauler": {
        "emoji": "🚚",
        "modes": {
            "Active": {
                0: {"name": "Haul Payload", "heat": 6.0, "wear": 0.15, "action_type": 1.0, "impact": 4.0},
                1: {"name": "Navigate", "heat": 3.0, "wear": 0.05, "action_type": 1.0, "impact": 1.0},
                2: {"name": "Thermal Vent", "heat": -12.0, "wear": -0.1, "action_type": 0.0, "impact": 0.0}
            },
            "Relay_Chrysalis": {  # Metamorphosis: Hardware static, acting as comms relay
                0: {"name": "Boost Network", "heat": 1.0, "wear": -0.2, "action_type": 1.0, "impact": 0.5},
                1: {"name": "Process Data", "heat": 0.5, "wear": -0.2, "action_type": 1.0, "impact": 0.2},
                2: {"name": "Deep Repair", "heat": -18.0, "wear": -0.5, "action_type": 0.0, "impact": 0.0}
            }
        }
    },
    "Borer": {
        "emoji": "🚜",
        "modes": {
            "Active": {
                0: {"name": "Deep Bore", "heat": 10.0, "wear": 0.3, "action_type": 2.0, "impact": 15.0},
                1: {"name": "Scan Vein", "heat": 2.0, "wear": 0.02, "action_type": 1.0, "impact": 1.0},
                2: {"name": "Emergency Cool", "heat": -15.0, "wear": -0.1, "action_type": 0.0, "impact": 0.0}
            },
            "Relay_Chrysalis": {
                0: {"name": "Seismic Ping", "heat": 1.0, "wear": -0.2, "action_type": 1.0, "impact": 0.5},
                1: {"name": "Idle", "heat": -2.0, "wear": -0.3, "action_type": 0.0, "impact": 0.0},
                2: {"name": "Deep Repair", "heat": -18.0, "wear": -0.5, "action_type": 0.0, "impact": 0.0}
            }
        }
    }
}


# ═════════════════════════════════════════════════════════════════════════════
#  4. STANDARDIZED SWARM AGENT (Cyber-Physical Node)
# ═════════════════════════════════════════════════════════════════════════════

class StandardizedSwarmNode(Node):
    def __init__(self, name: str, chassis: str, planet_id: str):
        super().__init__(f'{name}_{planet_id}')
        self.node_name = name
        self.chassis = chassis
        self.planet_id = planet_id

        self.config = CHASSIS_CONFIGS[chassis]
        self.emoji = self.config["emoji"]

        # Start in Active Software Mode
        self.software_mode = "Active"
        self.active_domain = self.config["modes"][self.software_mode]

        # Embodied State
        self.current_temp = 30.0  # Ambient starting
        self.previous_temp = 30.0
        self.machine_wear = 0.0

        self.planet_ambient_temp = 30.0
        self.local_danger_scent = 0.0
        self.local_opp_scent = 0.0

        self.last_reward = 0.0
        self.last_attention = 1.0
        self.last_mood = 0.0
        self.last_panic = 0.0
        self.last_action_taken = -1

        self.head = AdvancedNeuromorphicHead(n_actions=3)
        self.governor = SafetyGovernor(IndustrialThresholds(max_temp=90.0))  # Increased slightly for harsher planets
        self.perception = PerceptionPipeline(FeatureNormalizer(), HoloSynHeads(), StudentDistilledHeadsHF(),
                                             StudentDistilledHeadsBasic())

        # ROS 2 Pub/Sub
        self.pub_action = self.create_publisher(Float32MultiArray, f'/{self.planet_id}/agent_action', 10)
        self.create_subscription(Float32MultiArray, f'/{self.planet_id}/telemetry', self.telemetry_cb, 10)
        self.create_subscription(Float32MultiArray, f'/{self.planet_id}/stigmergy', self.stigmergy_cb, 10)

    def telemetry_cb(self, msg):
        self.planet_ambient_temp = msg.data[3]

    def stigmergy_cb(self, msg):
        self.local_danger_scent = msg.data[0]
        self.local_opp_scent = msg.data[1]

    def switch_software_mode(self, new_mode):
        """SOFTWARE METAMORPHOSIS: Retains standard physical chassis, alters logic constraints."""
        self.software_mode = new_mode
        self.active_domain = self.config["modes"][new_mode]

    def viral_synaptic_sync(self, peers):
        if not HAS_NEUROMORPHIC or self.last_reward < 1.5: return
        my_weights = np.array(self.head.S.w)
        for peer in peers:
            if peer.chassis == self.chassis and peer.node_name != self.node_name and peer.planet_id == self.planet_id:
                peer_weights = np.array(peer.head.S.w)
                peer.head.S.w = (peer_weights * 0.85) + (my_weights * 0.15)
                self.get_logger().info(f"✨ {self.node_name} virally synced weights to {peer.node_name}")

    def cognitive_loop(self, peers):
        # ── STANDARDIZED METAMORPHOSIS (Mode Shifting) ──
        if self.machine_wear > 4.5 and self.software_mode != "Relay_Chrysalis":
            self.get_logger().warn(
                f"🧬 {self.node_name} Critical Wear! Engaging Software Metamorphosis -> Relay Chrysalis Mode.")
            self.switch_software_mode("Relay_Chrysalis")
        elif self.machine_wear <= 0.5 and self.software_mode == "Relay_Chrysalis":
            self.get_logger().warn(f"🦋 {self.node_name} Diagnostics clear. Returning to Active Mode.")
            self.switch_software_mode("Active")

        bundle = ModalityBundle(
            E_t_aug=torch.randn(1, 528), E_i=torch.zeros(1, 2048),
            E_a=torch.randn(1, 2048), E_v=torch.randn(1, 2048),
            H=torch.zeros(1, 32, 3), feat_hf=torch.randn(1, 789),
            feat_basic=torch.randn(1, 5), mask=torch.ones(1, 4),
        )
        fused_emb, _, _ = self.perception.perceive(bundle)

        # Base cooling towards ambient planet temp
        if self.current_temp > self.planet_ambient_temp:
            self.current_temp -= 1.0

        actual_temp_delta = self.current_temp - self.previous_temp

        # Predictive Coding
        expected_delta = 0.0
        if self.last_action_taken in [0, 1, 2]:
            expected_delta = self.active_domain[self.last_action_taken]["heat"]
        elif self.last_action_taken == -1:
            expected_delta = -15.0

        mech_error_signal = 0.0
        if self.last_action_taken != -1:
            mech_error_signal = np.clip((actual_temp_delta - expected_delta) / 10.0, -1.0, 1.0)

            # Stigmergy Integration
        if self.local_danger_scent > 5.0: self.last_mood += 0.2
        if self.local_opp_scent > 5.0: self.last_attention += 0.5

        proposed_action = self.head.predict_and_learn(
            z=fused_emb, reward_signal=self.last_reward, attention_signal=self.last_attention,
            mood_signal=self.last_mood, mech_error_signal=mech_error_signal, panic_signal=self.last_panic
        )

        is_safe, reason = self.governor.validate_action(proposed_action, {"temperature": self.current_temp})
        self.previous_temp = self.current_temp
        self.last_attention = 1.0 + min(2.0, abs(actual_temp_delta) * 0.5)

        danger_spike = 0.0
        opp_spike = 0.0

        if is_safe:
            self.last_mood = -0.1
            self.last_panic = 0.0

            # Enhanced cooling reward to break loops
            self.last_reward = 1.0 if (self.current_temp / self.governor.limits.max_temp) < 0.70 else 0.2
            if actual_temp_delta < -2.0: self.last_reward += 0.8  # Strong reward for active cooling

            self.last_action_taken = proposed_action
            action_cfg = self.active_domain[proposed_action]

            self.current_temp += (action_cfg["heat"] + self.machine_wear)
            self.machine_wear = max(0.0, self.machine_wear + action_cfg["wear"])

            # Publish Economic Impact
            msg = Float32MultiArray(data=[action_cfg["action_type"], action_cfg["impact"], 0.0, 0.0])

            if action_cfg["action_type"] == 2.0:  # Mined something
                opp_spike = 2.0
                msg.data[3] = opp_spike

            self.pub_action.publish(msg)

            state_tag = f"[{self.software_mode[:3]}]"
            self.get_logger().info(
                f"{self.emoji} {self.node_name:<12} {state_tag} | {action_cfg['name']:<16} | Temp: {self.current_temp:>4.1f}C | Wear: {self.machine_wear:>3.1f} | Dopa: {self.last_reward:>4.1f}")

            self.viral_synaptic_sync(peers)
        else:
            self.last_mood = +0.5
            self.last_reward = 0.0
            self.last_panic = 1.0
            self.last_action_taken = -1

            danger_spike = 8.0
            msg = Float32MultiArray(data=[0.0, 0.0, danger_spike, 0.0])
            self.pub_action.publish(msg)

            self.get_logger().warn(
                f"{self.emoji} {self.node_name:<12} | BLOCKED! Noradrenaline injected. Danger Pheromone broadcast.")
            self.current_temp = max(self.planet_ambient_temp, self.current_temp - 20.0)  # Heavy venting


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT: MULTI-PLANETARY SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

def run_interplanetary_simulation():
    print("\n" + "═" * 80)
    print(" 🌌 PLANET FACTORY: INTERPLANETARY SWARM STANDARDIZATION")
    print("═" * 80)

    b2.start_scope()

    # 1. Initialize Networked Planetary Bodies
    # Asteroid_X is freezing, low wear. Volcan_Y is hot, high ambient wear.
    planet_x = PlanetaryBodyNode("Asteroid_X", ambient_temp=-10.0, init_materials=100.0)
    planet_y = PlanetaryBodyNode("Volcan_Y", ambient_temp=65.0, init_materials=500.0)

    # 2. Deploy Standardized Swarm Agents across planets
    swarm = [
        StandardizedSwarmNode("Drill-1", "Borer", "Asteroid_X"),
        StandardizedSwarmNode("Haul-1", "Hauler", "Asteroid_X"),

        StandardizedSwarmNode("Drill-2", "Borer", "Volcan_Y"),
        StandardizedSwarmNode("Drill-3", "Borer", "Volcan_Y")  # Swarm on a harsh planet
    ]

    print("\n🚀 COMMENCING MULTI-PLANETARY OPERATIONS...")
    for tick in range(1, 26):
        print(f"\n--- Cycle {tick} ---")

        # Process network events (Planet Physics & Stigmergy)
        planet_x.environment_tick()
        planet_y.environment_tick()

        # Agent AI ticks
        for insect in swarm:
            insect.cognitive_loop(peers=swarm)

        if tick % 5 == 0:
            print("-" * 40)
            planet_x.print_state()
            planet_y.print_state()
            print("-" * 40)

        time.sleep(0.05)

    print("\n🏁 INTERPLANETARY COLONIZATION COMPLETE.")
    print("═" * 80)


if __name__ == '__main__':
    if HAS_ROS2:
        print("Please run offline simulation for ecosystem test.")
    else:
        run_interplanetary_simulation()