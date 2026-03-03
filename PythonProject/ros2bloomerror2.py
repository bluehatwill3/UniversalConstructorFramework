"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PLANET FACTORY: PHASE 10 - GALACTIC EXPANSION ENGINE          ║
║                                                                              ║
║  Interplanetary Sovereign AI. Features Autogenous Node Spawning, Stigmergic  ║
║  Supply Chains, High-Latency Cargo Persistence, and Multi-Body Topology.     ║
║  Designed for Insectoid Swarm Deployment across Planets, Moons, and Comets.  ║
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
from typing import Dict, Any, List

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


# ── 🌐 ASYNCHRONOUS LATENCY-AWARE EVENT BUS (EXPANSION ENABLED) ──────────────
@dataclass
class NetworkPacket:
    topic: str
    payload: Any
    deliver_at: float
    source: str
    dest: str


class LatencyEventBus:
    """Models speed-of-light constraints and Interplanetary Cargo Logistics."""

    def __init__(self):
        self.topics = defaultdict(list)
        self.queue = deque()
        self.current_sim_time = 0.0

    def subscribe(self, topic, callback):
        self.topics[topic].append(callback)

    def publish(self, topic, msg, source_loc: str, dest_loc: str):
        # Latency based on Planetary Distance (Calculated for Reliability)
        latency = 0.01
        if source_loc != dest_loc and "GLOBAL" not in [source_loc, dest_loc]:
            # Reduced for simulation demo reliability, but physics-scaled
            latency = 2.0 if dest_loc == "Planet_Bloom" else 5.0

        packet = NetworkPacket(topic, msg, self.current_sim_time + latency, source_loc, dest_loc)
        self.queue.append(packet)
        self.queue = deque(sorted(self.queue, key=lambda x: x.deliver_at))

    def process_queue(self, current_time: float):
        self.current_sim_time = current_time
        delivered_packets = []
        while self.queue and self.queue[0].deliver_at <= current_time:
            packet = self.queue.popleft()
            delivered_packets.append(packet)
            for cb in self.topics[packet.topic]:
                cb(packet.payload, packet.source)
        return delivered_packets


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
            latency_network.subscribe(topic, lambda m, s: callback(m))
            return topic

        def create_publisher(self, msg_type, topic, qos, node_loc="GLOBAL"):
            class MockPub:
                def publish(self, msg, dest_loc="GLOBAL"):
                    latency_network.publish(topic, msg, node_loc, dest_loc)

            return MockPub()

        def create_timer(self, *args, **kwargs): pass


    class Twist:
        class Vector3:
            def __init__(self): self.x = 0.0; self.y = 0.0; self.z = 0.0

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
#  1. THE SNN BRAIN (With Viral Spore Syncing)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SNNNeuroConfig:
    tau_pre: float = 20.0
    tau_post: float = 20.0
    tau_c: float = 50.0
    tau_d: float = 120.0
    base_lr: float = 2.0
    w_decay: float = 0.01
    hetero_decay: float = 0.05
    mech_lr: float = 4.0
    panic_lr: float = 25.0
    synaptic_delay_max: float = 5.0


class AdvancedNeuromorphicHead(nn.Module):
    def __init__(self, n_actions: int, neuro_config: SNNNeuroConfig):
        super().__init__()
        self.n_actions = n_actions
        self.cfg = neuro_config
        if not HAS_NEUROMORPHIC: return

        b2.start_scope()
        self.P = b2.PoissonGroup(256, rates=np.zeros(256) * b2.Hz)
        self.G = b2.NeuronGroup(n_actions, 'dv/dt = (I - v) / (10*ms) : 1\nI : 1\nmood_modifier : 1',
                                threshold='v > (1.0 + mood_modifier)', reset='v=0', refractory=2 * b2.ms,
                                method='euler')

        syn_eqs = '''
        dApre/dt = -Apre / taupre : 1 (event-driven)
        dApost/dt = -Apost / taupost : 1 (event-driven)
        dc/dt = -c / tau_c : 1 (clock-driven)  
        dx/dt = (1 - x) / tau_d : 1 (clock-driven)
        dw/dt = (base_lr * attention) * c * reward - w_decay * w - hetero_decay * global_activity * w - mech_lr * mecherror * c - panic_lr * panicsignal * c : 1 (clock-driven)

        taupre : second (shared); taupost : second (shared); tau_c : second (shared); tau_d : second (shared)
        base_lr : 1/second (shared); reward : 1 (shared); attention : 1 (shared); w_decay : 1/second (shared)
        hetero_decay : 1/second (shared); global_activity : 1 (shared); mecherror : 1 (shared); mech_lr : 1/second (shared)
        panicsignal : 1 (shared); panic_lr : 1/second (shared)
        '''
        self.S = b2.Synapses(self.P, self.G, model=syn_eqs,
                             on_pre='v_post += w * x; x *= 0.8; Apre += 0.01; c += Apost',
                             on_post='Apost -= 0.01; c += Apre', method='euler')
        self.S.connect();
        self.S.w = 'rand() * 0.3';
        self.S.x = 1.0;
        self.S.delay = f'rand() * {self.cfg.synaptic_delay_max} * ms'

        self.S.taupre = self.cfg.tau_pre * b2.ms;
        self.S.taupost = self.cfg.tau_post * b2.ms;
        self.S.tau_c = self.cfg.tau_c * b2.ms;
        self.S.tau_d = self.cfg.tau_d * b2.ms
        self.S.base_lr = self.cfg.base_lr * b2.Hz;
        self.S.w_decay = self.cfg.w_decay * b2.Hz;
        self.S.hetero_decay = self.cfg.hetero_decay * b2.Hz
        self.S.mech_lr = self.cfg.mech_lr * b2.Hz;
        self.S.panic_lr = self.cfg.panic_lr * b2.Hz
        self.S.reward = 0.0;
        self.S.attention = 1.0;
        self.G.mood_modifier = 0.0
        self.M = b2.SpikeMonitor(self.G);
        self.net = b2.Network(self.P, self.G, self.S, self.M)
        self.last_counts = np.zeros(n_actions)

    def predict_and_learn(self, z, reward_signal, attention_signal, mood_signal, mech_error_signal,
                          panic_signal=0.0) -> tuple:
        if not HAS_NEUROMORPHIC: return 0, 0.0
        t_start = time.perf_counter()
        self.P.rates = np.clip(z.squeeze().cpu().numpy(), 0, 1) * 100 * b2.Hz
        self.S.reward, self.S.attention, self.S.mecherror, self.S.panicsignal = reward_signal, attention_signal, mech_error_signal, panic_signal
        self.S.global_activity = float(np.sum(self.last_counts) / max(1, self.n_actions))
        self.G.mood_modifier = mood_signal
        self.net.run(20 * b2.ms)
        step_counts = np.array(self.M.count) - self.last_counts
        self.last_counts = np.array(self.M.count)
        return (int(np.argmax(step_counts)) if np.sum(step_counts) > 0 else np.random.randint(0, self.n_actions)), (
                    time.perf_counter() - t_start)


# ═════════════════════════════════════════════════════════════════════════════
#  2. PLANETARY NODES (Supply Chain & Autogenous Expansion)
# ═════════════════════════════════════════════════════════════════════════════

class PlanetaryBodyNode(Node):
    def __init__(self, planet_id: str, ambient_temp: float, is_destination=False):
        super().__init__(f'planet_{planet_id}')
        self.planet_id, self.ambient_temp, self.is_destination = planet_id, ambient_temp, is_destination
        self.energy, self.raw_materials, self.finished_goods, self.bloom_delivered = 5000.0, 50.0, 0.0, 0.0
        self.infrastructure = 1.0

        self.pub_tele = self.create_publisher(Float32MultiArray, f'/{self.planet_id}/telemetry', 10,
                                              node_loc=self.planet_id)
        self.create_subscription(Float32MultiArray, f'/{self.planet_id}/agent_action', self.agent_action_cb, 10)
        self.create_subscription(Float32MultiArray, f'/global/logistics', self.logistics_cb, 10)

    def logistics_cb(self, msg):
        cargo_type, amount = msg.data
        if self.is_destination:
            self.bloom_delivered += amount
        else:
            self.finished_goods += amount

    def agent_action_cb(self, msg):
        # [action_type, energy_cost, mats_delta, goods_delta, expansion_flag]
        atype, cost, m_delta, g_delta, expand = msg.data
        self.energy -= cost
        self.raw_materials += m_delta

        # Production Logic
        if g_delta > 0 and self.raw_materials >= 10.0:
            self.raw_materials -= 10.0
            self.finished_goods += 1.0

        # Expansion Logic
        if expand > 0 and self.raw_materials >= 100.0:
            self.raw_materials -= 100.0
            self.infrastructure += 1.0
            # Broadcast Spawning Signal to the Tournament Loop
            latency_network.publish('/global/spawn', Float32MultiArray(data=[1.0]), self.planet_id, "GLOBAL")

        # Launch Logic
        if atype == 4.0 and self.finished_goods >= 1.0:
            self.finished_goods -= 1.0
            latency_network.publish('/global/logistics', Float32MultiArray(data=[1.0, 1.0]), self.planet_id,
                                    "Planet_Bloom")

    def environment_tick(self):
        # Telemetry: [Energy, Mats, Goods, Ambient, Infra]
        self.pub_tele.publish(Float32MultiArray(
            data=[self.energy, self.raw_materials, self.finished_goods, self.ambient_temp, self.infrastructure]))

    def print_state(self):
        role = "DESTINATION" if self.is_destination else "BASE"
        print(
            f"🌍 [{self.planet_id.upper()} ({role})] Infra: {self.infrastructure:.0f} | Mats: {self.raw_materials:.0f} | Goods: {self.finished_goods:.0f} | BLOOM: {self.bloom_delivered:.0f}")


# ═════════════════════════════════════════════════════════════════════════════
#  3. CHASSIS MODES (Refining & Spawning)
# ═════════════════════════════════════════════════════════════════════════════

CHASSIS_CONFIGS = {
    "Hauler": {
        "emoji": "🚚",
        "modes": {
            "Active": {
                0: {"name": "Refine Ore", "heat": 8.0, "wear": 0.1, "energy": 15.0, "goods": 1.0, "type": 1.0,
                    "expand": 0.0},
                1: {"name": "Bloom Launch", "heat": 25.0, "wear": 0.5, "energy": 500.0, "goods": 0.0, "type": 4.0,
                    "expand": 0.0},
                2: {"name": "Thermal Vent", "heat": -15.0, "wear": -0.1, "energy": 0.0, "goods": 0.0, "type": 0.0,
                    "expand": 0.0}
            },
            "Relay_Chrysalis": {
                2: {"name": "Repair/Idle", "heat": -20.0, "wear": -0.5, "energy": 0.0, "goods": 0.0, "type": 0.0,
                    "expand": 0.0},
                0: {"name": "Idle", "heat": -1.0, "wear": -0.1, "energy": 0.0, "goods": 0.0, "type": 0.0,
                    "expand": 0.0},
                1: {"name": "Idle", "heat": -1.0, "wear": -0.1, "energy": 0.0, "goods": 0.0, "type": 0.0, "expand": 0.0}
            }
        }
    },
    "Borer": {
        "emoji": "🚜",
        "modes": {
            "Active": {
                0: {"name": "Deep Bore", "heat": 15.0, "wear": 0.3, "energy": 15.0, "mats": 20.0, "type": 2.0,
                    "expand": 0.0},
                1: {"name": "Spawn Swarm Node", "heat": 10.0, "wear": 0.2, "energy": 200.0, "mats": 0.0, "type": 0.0,
                    "expand": 1.0},
                2: {"name": "Emergency Cool", "heat": -15.0, "wear": -0.1, "energy": 0.0, "mats": 0.0, "type": 0.0,
                    "expand": 0.0}
            },
            "Relay_Chrysalis": {
                2: {"name": "Deep Repair", "heat": -20.0, "wear": -0.5, "energy": 0.0, "mats": 0.0, "type": 0.0,
                    "expand": 0.0},
                0: {"name": "Idle", "heat": 0, "wear": 0, "energy": 0, "mats": 0, "type": 0, "expand": 0},
                1: {"name": "Idle", "heat": 0, "wear": 0, "energy": 0, "mats": 0, "type": 0, "expand": 0}
            }
        }
    }
}


# ═════════════════════════════════════════════════════════════════════════════
#  4. THE AGENT (Expands and Viral Syncs)
# ═════════════════════════════════════════════════════════════════════════════

class StandardizedSwarmNode(Node):
    def __init__(self, name: str, chassis: str, planet_id: str, neuro_cfg: SNNNeuroConfig):
        super().__init__(f'{name}_{planet_id}')
        self.node_name, self.chassis, self.planet_id = name, chassis, planet_id
        self.config = CHASSIS_CONFIGS[chassis]
        self.software_mode = "Active"
        self.active_domain = self.config["modes"][self.software_mode]

        self.current_temp, self.previous_temp, self.machine_wear = 30.0, 30.0, 0.0
        self.last_reward, self.last_attention, self.last_mood, self.last_panic, self.last_action_taken = 0.0, 1.0, 0.0, 0.0, -1
        self.planet_tele = [0.0, 0.0, 0.0, 30.0, 1.0]  # Energy, Mats, Goods, Ambient, Infra

        self.head = AdvancedNeuromorphicHead(n_actions=3, neuro_config=neuro_cfg)
        self.governor = SafetyGovernor(IndustrialThresholds(max_temp=95.0))
        self.perception = PerceptionPipeline(FeatureNormalizer(), HoloSynHeads(), StudentDistilledHeadsHF(),
                                             StudentDistilledHeadsBasic())

        self.pub_action = self.create_publisher(Float32MultiArray, f'/{self.planet_id}/agent_action', 10,
                                                node_loc=self.planet_id)
        self.create_subscription(Float32MultiArray, f'/{self.planet_id}/telemetry', self.telemetry_cb, 10)

    def telemetry_cb(self, msg):
        self.planet_tele = msg.data

    def cognitive_loop(self):
        # Standard Metamorphosis Logic
        if self.machine_wear > 4.5 and self.software_mode != "Relay_Chrysalis":
            self.software_mode = "Relay_Chrysalis";
            self.active_domain = self.config["modes"]["Relay_Chrysalis"]
        elif self.machine_wear <= 0.5 and self.software_mode == "Relay_Chrysalis":
            self.software_mode = "Active";
            self.active_domain = self.config["modes"]["Active"]

        bundle = ModalityBundle(E_t_aug=torch.randn(1, 528), E_i=torch.zeros(1, 2048), E_a=torch.randn(1, 2048),
                                E_v=torch.randn(1, 2048), H=torch.zeros(1, 32, 3), feat_hf=torch.randn(1, 789),
                                feat_basic=torch.randn(1, 5), mask=torch.ones(1, 4))
        fused_emb, _, _ = self.perception.perceive(bundle)

        actual_temp_delta = self.current_temp - self.previous_temp
        proposed_action, compute_time = self.head.predict_and_learn(z=fused_emb, reward_signal=self.last_reward,
                                                                    attention_signal=self.last_attention,
                                                                    mood_signal=self.last_mood, mech_error_signal=0.0,
                                                                    panic_signal=self.last_panic)

        is_safe, reason = self.governor.validate_action(proposed_action, {"temperature": self.current_temp})
        self.previous_temp = self.current_temp

        if is_safe:
            self.last_mood, self.last_panic = -0.1, 0.0
            self.last_reward = 1.2 if (self.current_temp / 95.0) < 0.70 else 0.2
            self.last_action_taken = proposed_action
            action_cfg = self.active_domain[proposed_action]

            # Physics
            self.current_temp += action_cfg["heat"]
            self.machine_wear = max(0.0, self.machine_wear + action_cfg["wear"])

            # Topic: [action_type, cost, material_delta, goods_delta, expansion_flag]
            self.pub_action.publish(Float32MultiArray(
                data=[action_cfg.get("type", 0.0), action_cfg["energy"], action_cfg.get("mats", 0.0),
                      action_cfg.get("goods", 0.0), action_cfg.get("expand", 0.0)]), dest_loc=self.planet_id)

            self.get_logger().info(
                f"{self.node_name:<10} | {action_cfg['name']:<18} | Temp: {self.current_temp:>4.1f}C | Bloom-Ready: {self.planet_tele[2]:.0f}")
        else:
            self.last_mood, self.last_reward, self.last_panic, self.last_action_taken = 0.5, 0.0, 1.0, -1
            self.current_temp = max(self.planet_tele[3], self.current_temp - 20.0)


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT: BLOOM COLONIZATION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def run_bloom_colonization():
    print("\n" + "═" * 80)
    print(" 🌌 PLANET FACTORY: GALACTIC EXPANSION ENGINE (BLOOM LOGISTICS)")
    print("═" * 80)
    b2.start_scope()

    # 1. Initialize Network
    bloom = PlanetaryBodyNode("Planet_Bloom", ambient_temp=25.0, is_destination=True)
    mine_a = PlanetaryBodyNode("Asteroid_A1", ambient_temp=-40.0)
    mine_b = PlanetaryBodyNode("Moon_B2", ambient_temp=-10.0)

    hyper_cfg = SNNNeuroConfig(base_lr=3.0)

    swarm = [
        StandardizedSwarmNode("Borer-1", "Borer", "Asteroid_A1", hyper_cfg),
        StandardizedSwarmNode("Hauler-1", "Hauler", "Asteroid_A1", hyper_cfg),
        StandardizedSwarmNode("Borer-2", "Borer", "Moon_B2", hyper_cfg)
    ]

    # Track Spawning events
    def spawn_handler(msg, source):
        new_id = f"Borer-Gen{len(swarm) + 1}"
        swarm.append(StandardizedSwarmNode(new_id, "Borer", source, hyper_cfg))
        print(f"✨ ✨ ✨ AUTOGENOUS EXPANSION: New agent {new_id} spawned on {source}!")

    latency_network.subscribe('/global/spawn', spawn_handler)

    print("\n🚀 COMMENCING LOGISTICS & EXPANSION LOOP...")
    sim_clock = 0.0
    for cycle in range(1, 61):  # Increased cycles for latency survival
        sim_clock += 0.1
        latency_network.process_queue(sim_clock)

        bloom.environment_tick();
        mine_a.environment_tick();
        mine_b.environment_tick()

        for insect in swarm: insect.cognitive_loop()

        if cycle % 15 == 0:
            print("-" * 50)
            bloom.print_state();
            mine_a.print_state();
            mine_b.print_state()
            print("-" * 50)
        time.sleep(0.01)

    print("\n🏁 INTERPLANETARY MISSION END.")
    bloom.print_state()
    print("═" * 80)


if __name__ == '__main__':
    run_bloom_colonization()