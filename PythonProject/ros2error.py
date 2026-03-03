"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PLANET FACTORY: PHASE 8 - ASYNCHRONOUS COMPLEXITY             ║
║                                                                              ║
║  Unrestricted, deeply customizable architecture. Sacrifices ease-of-use for  ║
║  raw power. Features True Time (Cognitive Lag), Speed-of-Light Network       ║
║  Latency, Hardware Constraint Solving, and Deep-Parameter Neural Exposure.   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Any

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


# ── 🌐 ASYNCHRONOUS LATENCY-AWARE EVENT BUS ──────────────────────────────────
@dataclass
class NetworkPacket:
    topic: str
    payload: Any
    deliver_at: float  # Absolute simulation time when this packet arrives


class LatencyEventBus:
    """Models speed-of-light constraints and network bandwidth."""

    def __init__(self):
        self.topics = defaultdict(list)
        self.queue = deque()
        self.current_sim_time = 0.0

    def subscribe(self, topic, callback):
        self.topics[topic].append(callback)

    def publish(self, topic, msg, source_loc: str, dest_loc: str):
        # Calculate latency based on distance (Interplanetary vs Local)
        latency = 0.01  # 10ms local
        if source_loc != dest_loc and source_loc != "GLOBAL":
            latency = 15.0  # 15 seconds interplanetary transmission delay

        packet = NetworkPacket(topic, msg, self.current_sim_time + latency)
        self.queue.append(packet)
        # Sort queue by delivery time (simplistic scheduler)
        self.queue = deque(sorted(self.queue, key=lambda x: x.deliver_at))

    def process_queue(self, current_time: float):
        self.current_sim_time = current_time
        while self.queue and self.queue[0].deliver_at <= current_time:
            packet = self.queue.popleft()
            for cb in self.topics[packet.topic]:
                cb(packet.payload)


latency_network = LatencyEventBus()

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
            latency_network.subscribe(topic, callback)
            return topic

        def create_publisher(self, msg_type, topic, qos, node_loc="GLOBAL"):
            class MockPub:
                def publish(self, msg, dest_loc="GLOBAL"):
                    latency_network.publish(topic, msg, node_loc, dest_loc)

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
        pass; class


    Image:
    pass;


    class LaserScan:
        pass; class


    Imu:
    pass


    class JointState:
        pass; class


    WrenchStamped:
    pass;


    class JointTrajectory:
        pass


    class JointTrajectoryPoint:
        pass; class


    CvBridge:
    pass


# ═════════════════════════════════════════════════════════════════════════════
#  1. UNRESTRICTED SNN (Deep Parameter Exposure & Spatial Delays)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SNNNeuroConfig:
    """Hyper-parameter exposure for full biological customization."""
    tau_pre: float = 20.0  # ms
    tau_post: float = 20.0  # ms
    tau_c: float = 50.0  # ms (Eligibility trace decay)
    tau_d: float = 120.0  # ms (Short-term depression recovery)
    base_lr: float = 1.8  # Hz
    w_decay: float = 0.02  # Hz (Homeostasis)
    hetero_decay: float = 0.05  # Hz
    mech_lr: float = 3.5  # Hz (Predictive Coding)
    panic_lr: float = 20.0  # Hz (Noradrenaline)
    synaptic_delay_max: float = 3.0  # ms (Spatial modeling)


class AdvancedNeuromorphicHead(nn.Module):
    def __init__(self, n_actions: int, neuro_config: SNNNeuroConfig):
        super().__init__()
        self.n_actions = n_actions
        self.cfg = neuro_config

        if not HAS_NEUROMORPHIC: return

        self.P = b2.PoissonGroup(256, rates=np.zeros(256) * b2.Hz)

        eqs_neurons = '''
        dv/dt = (I - v) / (10*ms) : 1
        I : 1
        mood_modifier : 1  
        '''
        self.G = b2.NeuronGroup(n_actions, eqs_neurons, threshold='v > (1.0 + mood_modifier)', reset='v=0',
                                refractory=2 * b2.ms, method='euler')

        syn_eqs = '''
        dApre/dt = -Apre / taupre : 1 (event-driven)
        dApost/dt = -Apost / taupost : 1 (event-driven)
        dc/dt = -c / tau_c : 1 (clock-driven)  
        dx/dt = (1 - x) / tau_d : 1 (clock-driven)

        # Fully unrestricted multi-modulator equation
        dw/dt = (base_lr * attention) * c * reward - w_decay * w - hetero_decay * global_activity * w - mech_lr * mecherror * c - panic_lr * panicsignal * c : 1 (clock-driven)

        taupre : second (shared)
        taupost : second (shared)
        tau_c : second (shared)
        tau_d : second (shared)
        base_lr : 1/second (shared)
        reward : 1 (shared)
        attention : 1 (shared)
        w_decay : 1/second (shared)
        hetero_decay : 1/second (shared)
        global_activity : 1 (shared)
        mecherror : 1 (shared)
        mech_lr : 1/second (shared)
        panicsignal : 1 (shared)
        panic_lr : 1/second (shared)
        '''

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

        # Inject Spatial Synaptic Delays
        self.S.delay = f'rand() * {self.cfg.synaptic_delay_max} * ms'

        # Map Deep Configuration
        self.S.taupre = self.cfg.tau_pre * b2.ms
        self.S.taupost = self.cfg.tau_post * b2.ms
        self.S.tau_c = self.cfg.tau_c * b2.ms
        self.S.tau_d = self.cfg.tau_d * b2.ms
        self.S.base_lr = self.cfg.base_lr * b2.Hz
        self.S.w_decay = self.cfg.w_decay * b2.Hz
        self.S.hetero_decay = self.cfg.hetero_decay * b2.Hz
        self.S.mech_lr = self.cfg.mech_lr * b2.Hz
        self.S.panic_lr = self.cfg.panic_lr * b2.Hz

        self.S.global_activity = 0.0
        self.S.mecherror = 0.0
        self.S.panicsignal = 0.0
        self.S.reward = 0.0
        self.S.attention = 1.0

        self.G.mood_modifier = 0.0
        self.M = b2.SpikeMonitor(self.G)
        self.net = b2.Network(self.P, self.G, self.S, self.M)
        self.last_counts = np.zeros(n_actions)

    def predict_and_learn(self, z, reward_signal, attention_signal, mood_signal, mech_error_signal,
                          panic_signal=0.0) -> tuple:
        if not HAS_NEUROMORPHIC: return 0, 0.0

        t_start = time.perf_counter()  # ── LATENCY PROFILING ──

        self.P.rates = np.clip(z.squeeze().cpu().numpy(), 0, 1) * 100 * b2.Hz

        self.S.reward = reward_signal
        self.S.attention = attention_signal
        self.S.mecherror = mech_error_signal
        self.S.panicsignal = panic_signal
        self.S.global_activity = float(np.sum(self.last_counts) / max(1, self.n_actions))
        self.G.mood_modifier = mood_signal

        self.net.run(20 * b2.ms)

        current_counts = np.array(self.M.count)
        step_counts = current_counts - self.last_counts
        self.last_counts = current_counts.copy()

        compute_time = time.perf_counter() - t_start

        action = int(np.argmax(step_counts)) if np.sum(step_counts) > 0 else np.random.randint(0, self.n_actions)
        return action, compute_time


# ═════════════════════════════════════════════════════════════════════════════
#  2. PHYSICS & KINEMATICS CONSTRAINT SOLVER
# ═════════════════════════════════════════════════════════════════════════════

class PhysicsConstraintSolver:
    """We don't assume limits; we enforce physics."""

    def __init__(self, chassis: str):
        self.max_thermal_dissipation = 20.0  # Max heat lost per second
        self.thermal_mass = 50.0 if chassis == "Borer" else 30.0  # Higher mass = slower to heat up, slower to cool
        self.kinematic_latency = 0.5  # Seconds it takes for actuators to spool up

    def calculate_physical_reaction(self, requested_heat: float, current_temp: float, ambient: float) -> float:
        # Calculates true delta T taking into account thermal mass and ambient gradients
        heat_gradient = (current_temp - ambient) * 0.05  # Natural cooling based on gradient
        true_delta = (requested_heat / self.thermal_mass) * 10.0 - heat_gradient
        return true_delta


# ═════════════════════════════════════════════════════════════════════════════
#  3. DOMAIN CONFIGURATIONS & NETWORKED PLANET
# ═════════════════════════════════════════════════════════════════════════════

CHASSIS_CONFIGS = {
    "Hauler": {
        "emoji": "🚚",
        "modes": {
            "Active": {
                0: {"name": "Haul Payload", "heat": 8.0, "wear": 0.15, "energy": 4.0},
                1: {"name": "Navigate", "heat": 3.0, "wear": 0.05, "energy": 1.0},
                2: {"name": "Thermal Vent", "heat": -15.0, "wear": -0.1, "energy": 0.0}
            },
            "Relay_Chrysalis": {
                0: {"name": "Boost Network", "heat": 1.0, "wear": -0.2, "energy": 0.5},
                1: {"name": "Process Data", "heat": 0.5, "wear": -0.2, "energy": 0.2},
                2: {"name": "Deep Repair", "heat": -20.0, "wear": -0.5, "energy": 0.0}
            }
        }
    },
    "Borer": {
        "emoji": "🚜",
        "modes": {
            "Active": {
                0: {"name": "Deep Bore", "heat": 15.0, "wear": 0.3, "energy": 15.0},
                1: {"name": "Scan Vein", "heat": 2.0, "wear": 0.02, "energy": 1.0},
                2: {"name": "Emergency Cool", "heat": -15.0, "wear": -0.1, "energy": 0.0}
            },
            "Relay_Chrysalis": {
                0: {"name": "Seismic Ping", "heat": 1.0, "wear": -0.2, "energy": 0.5},
                1: {"name": "Idle", "heat": -2.0, "wear": -0.3, "energy": 0.0},
                2: {"name": "Deep Repair", "heat": -20.0, "wear": -0.5, "energy": 0.0}
            }
        }
    }
}


class PlanetaryBodyNode(Node):
    def __init__(self, planet_id: str, ambient_temp: float):
        super().__init__(f'planet_{planet_id}')
        self.planet_id = planet_id
        self.ambient_temp = ambient_temp
        self.global_energy = 5000.0
        self.pheromones = {"danger": 0.0, "opportunity": 0.0}

        self.pub_telemetry = self.create_publisher(Float32MultiArray, f'/{self.planet_id}/telemetry', 10,
                                                   node_loc=self.planet_id)
        self.pub_stigmergy = self.create_publisher(Float32MultiArray, f'/{self.planet_id}/stigmergy', 10,
                                                   node_loc=self.planet_id)
        self.create_subscription(Float32MultiArray, f'/{self.planet_id}/agent_action', self.agent_action_callback, 10)

    def agent_action_callback(self, msg):
        self.global_energy -= msg.data[0]  # Energy impact
        self.pheromones["danger"] += msg.data[1]
        self.pheromones["opportunity"] += msg.data[2]

    def environment_tick(self):
        self.pheromones["danger"] = max(0.0, self.pheromones["danger"] - 1.5)
        self.pheromones["opportunity"] = max(0.0, self.pheromones["opportunity"] - 0.5)
        self.pub_telemetry.publish(Float32MultiArray(data=[self.global_energy, self.ambient_temp]),
                                   dest_loc=self.planet_id)
        self.pub_stigmergy.publish(Float32MultiArray(data=[self.pheromones["danger"], self.pheromones["opportunity"]]),
                                   dest_loc=self.planet_id)

    def print_state(self):
        print(
            f"🌍 [{self.planet_id.upper()}] Energy: {self.global_energy:.0f} | Danger: {self.pheromones['danger']:.1f}")


# ═════════════════════════════════════════════════════════════════════════════
#  4. THE ASYNCHRONOUS AGENT (Handles True Time & Cognitive Lag)
# ═════════════════════════════════════════════════════════════════════════════

class StandardizedSwarmNode(Node):
    def __init__(self, name: str, chassis: str, planet_id: str, neuro_cfg: SNNNeuroConfig):
        super().__init__(f'{name}_{planet_id}')
        self.node_name = name
        self.chassis = chassis
        self.planet_id = planet_id
        self.config = CHASSIS_CONFIGS[chassis]
        self.emoji = self.config["emoji"]

        self.software_mode = "Active"
        self.active_domain = self.config["modes"][self.software_mode]
        self.physics = PhysicsConstraintSolver(chassis)

        self.current_temp = 30.0
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

        self.head = AdvancedNeuromorphicHead(n_actions=3, neuro_config=neuro_cfg)
        self.governor = SafetyGovernor(IndustrialThresholds(max_temp=95.0))
        self.perception = PerceptionPipeline(FeatureNormalizer(), HoloSynHeads(), StudentDistilledHeadsHF(),
                                             StudentDistilledHeadsBasic())

        self.pub_action = self.create_publisher(Float32MultiArray, f'/{self.planet_id}/agent_action', 10,
                                                node_loc=self.planet_id)
        self.pub_sync = self.create_publisher(Float32MultiArray, f'/global/synaptic_sync', 10, node_loc=self.planet_id)

        self.create_subscription(Float32MultiArray, f'/{self.planet_id}/telemetry', self.telemetry_cb, 10)
        self.create_subscription(Float32MultiArray, f'/{self.planet_id}/stigmergy', self.stigmergy_cb, 10)
        self.create_subscription(Float32MultiArray, f'/global/synaptic_sync', self.sync_cb, 10)

    def telemetry_cb(self, msg):
        self.planet_ambient_temp = msg.data[1]

    def stigmergy_cb(self, msg):
        self.local_danger_scent = msg.data[0]; self.local_opp_scent = msg.data[1]

    def sync_cb(self, msg):
        # Asynchronous weight sync received
        sender, weights = msg.data[0], msg.data[1:]
        if sender != float(hash(self.node_name)) and self.last_reward < 1.5:
            peer_weights = np.array(weights)
            self.head.S.w = (np.array(self.head.S.w) * 0.85) + (peer_weights * 0.15)

    def switch_software_mode(self, new_mode):
        self.software_mode = new_mode
        self.active_domain = self.config["modes"][new_mode]

    def cognitive_loop(self):
        if self.machine_wear > 4.5 and self.software_mode != "Relay_Chrysalis":
            self.get_logger().warn(f"🧬 {self.node_name} Critical Wear! Metamorphosis -> Relay Chrysalis.")
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

        actual_temp_delta = self.current_temp - self.previous_temp
        expected_delta = 0.0
        if self.last_action_taken != -1:
            raw_heat = self.active_domain[self.last_action_taken]["heat"]
            expected_delta = self.physics.calculate_physical_reaction(raw_heat, self.current_temp,
                                                                      self.planet_ambient_temp)

        mech_error_signal = 0.0
        if self.last_action_taken != -1:
            mech_error_signal = np.clip((actual_temp_delta - expected_delta) / 10.0, -1.0, 1.0)

        if self.local_danger_scent > 5.0: self.last_mood += 0.3
        if self.local_opp_scent > 5.0: self.last_attention += 0.5

        proposed_action, compute_time = self.head.predict_and_learn(
            z=fused_emb, reward_signal=self.last_reward, attention_signal=self.last_attention,
            mood_signal=self.last_mood, mech_error_signal=mech_error_signal, panic_signal=self.last_panic
        )

        # ── COGNITIVE LAG PENALTY ──
        # If the brain takes too long (e.g. heavy computation > 0.05s threshold), the world moves on.
        lag_penalty = 0.0
        if compute_time > 0.05:
            lag_penalty = (compute_time - 0.05) * 10.0  # Heat builds up while thinking
            self.current_temp += lag_penalty

        is_safe, reason = self.governor.validate_action(proposed_action, {"temperature": self.current_temp})
        self.previous_temp = self.current_temp
        self.last_attention = 1.0 + min(2.0, abs(actual_temp_delta) * 0.5)

        if is_safe:
            self.last_mood = -0.1
            self.last_panic = 0.0

            self.last_reward = 1.0 if (self.current_temp / self.governor.limits.max_temp) < 0.70 else 0.2
            if actual_temp_delta < -2.0: self.last_reward += 0.8

            self.last_action_taken = proposed_action
            action_cfg = self.active_domain[proposed_action]

            # Request heat vs Physical reality
            physical_delta = self.physics.calculate_physical_reaction(action_cfg["heat"] + self.machine_wear,
                                                                      self.current_temp, self.planet_ambient_temp)
            self.current_temp += physical_delta
            self.machine_wear = max(0.0, self.machine_wear + action_cfg["wear"])

            msg = Float32MultiArray(data=[action_cfg["energy"], 0.0, 0.0])
            if action_cfg["heat"] > 10.0: msg.data[2] = 2.0  # Opportunity spike for heavy work
            self.pub_action.publish(msg, dest_loc=self.planet_id)

            state_tag = f"[{self.software_mode[:3]}]"
            lag_str = f"| Lag: {compute_time * 1000:.0f}ms"
            self.get_logger().info(
                f"{self.emoji} {self.node_name:<12} {state_tag} | {action_cfg['name']:<16} | Temp: {self.current_temp:>4.1f}C | Wear: {self.machine_wear:>3.1f} {lag_str}")

            if self.last_reward > 1.5:
                # Async broadcast weights (Interplanetary destination = "GLOBAL")
                sync_payload = [float(hash(self.node_name))] + list(np.array(self.head.S.w))
                self.pub_sync.publish(Float32MultiArray(data=sync_payload), dest_loc="GLOBAL")
        else:
            self.last_mood = +0.5
            self.last_reward = 0.0
            self.last_panic = 1.0
            self.last_action_taken = -1

            msg = Float32MultiArray(data=[0.0, 8.0, 0.0])  # Emitting Danger
            self.pub_action.publish(msg, dest_loc=self.planet_id)

            self.get_logger().warn(
                f"{self.emoji} {self.node_name:<12} | BLOCKED! Constraints hit. Danger Pheromone broadcast.")
            self.current_temp = max(self.planet_ambient_temp, self.current_temp - 20.0)

        # ═════════════════════════════════════════════════════════════════════════════


#  ENTRY POINT: ASYNCHRONOUS EVENT ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def run_asynchronous_engine():
    print("\n" + "═" * 80)
    print(" 🌌 PLANET FACTORY: UNRESTRICTED ASYNCHRONOUS ARCHITECTURE")
    print("═" * 80)

    b2.start_scope()

    planet_x = PlanetaryBodyNode("Asteroid_X", ambient_temp=-20.0)
    planet_y = PlanetaryBodyNode("Volcan_Y", ambient_temp=75.0)

    # Deep parameter injection
    hyper_cfg = SNNNeuroConfig(tau_pre=25.0, base_lr=2.5, panic_lr=25.0, synaptic_delay_max=5.0)

    swarm = [
        StandardizedSwarmNode("Drill-1", "Borer", "Asteroid_X", hyper_cfg),
        StandardizedSwarmNode("Haul-1", "Hauler", "Asteroid_X", hyper_cfg),
        StandardizedSwarmNode("Drill-2", "Borer", "Volcan_Y", hyper_cfg)
    ]

    print("\n🚀 COMMENCING ASYNCHRONOUS OPERATIONS...")
    sim_clock = 0.0
    tick_interval = 0.1  # 100ms real-time slice

    for cycle in range(1, 26):
        sim_clock += tick_interval
        print(f"\n--- T+{sim_clock:.1f}s ---")

        # Process network latency queue (Speed of light delays)
        latency_network.process_queue(sim_clock)

        planet_x.environment_tick()
        planet_y.environment_tick()

        for insect in swarm:
            insect.cognitive_loop()

        if cycle % 5 == 0:
            print("-" * 40)
            planet_x.print_state()
            planet_y.print_state()
            print("-" * 40)

    print("\n🏁 SIMULATION COMPLETE.")
    print("═" * 80)


if __name__ == '__main__':
    if HAS_ROS2:
        print("Please run offline simulation for ecosystem test.")
    else:
        run_asynchronous_engine()