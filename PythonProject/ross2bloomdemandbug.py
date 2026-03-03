"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PLANET FACTORY: PHASE 9 - BLOOM LOGISTICS & EXPANSION         ║
║                                                                              ║
║  Interplanetary Supply Chain. Features Multi-Stage Production (Ore -> Goods),║
║  Delta-V Physics Constraints, Autogenous Expansion (Node Spawning), and      ║
║  Targeted Delivery to "Planet Bloom" across high-latency orbits.             ║
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


# ── 🌐 ASYNCHRONOUS LATENCY-AWARE EVENT BUS (CARGO SUPPORTED) ────────────────
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
        # Latency based on Planetary Distance
        latency = 0.01
        if source_loc != dest_loc and "GLOBAL" not in [source_loc, dest_loc]:
            # Simulate 15s to 60s Interplanetary Flight/Transmission
            latency = 15.0 if dest_loc == "Planet_Bloom" else 30.0

        packet = NetworkPacket(topic, msg, self.current_sim_time + latency, source_loc, dest_loc)
        self.queue.append(packet)
        self.queue = deque(sorted(self.queue, key=lambda x: x.deliver_at))

    def process_queue(self, current_time: float):
        self.current_sim_time = current_time
        while self.queue and self.queue[0].deliver_at <= current_time:
            packet = self.queue.popleft()
            for cb in self.topics[packet.topic]:
                cb(packet.payload, packet.source)


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
            # Wrapper to handle the (msg, source) callback of our LatencyBus
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


    # Corrected Mock Syntax
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
#  1. UNRESTRICTED SNN (Cognitive Lag & Spatial Scaling)
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
                             on_pre='v_post += w * x; x *= 0.8; Apre += 0.01; c += Apost',
                             on_post='Apost -= 0.01; c += Apre', method='euler')
        self.S.connect();
        self.S.w = 'rand() * 0.3';
        self.S.x = 1.0;
        self.S.delay = f'rand() * {self.cfg.synaptic_delay_max} * ms'

        # Mapping constants
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
#  2. PHYSICS & KINEMATICS (Delta-V & Orbital Energy)
# ═════════════════════════════════════════════════════════════════════════════

class PhysicsConstraintSolver:
    def __init__(self, chassis: str):
        self.thermal_mass = 50.0 if chassis == "Borer" else 30.0
        # Interplanetary ultimatum: Expending energy for mass transport
        self.launch_energy_cost = 500.0 if chassis == "Hauler" else 1000.0

    def calculate_physical_reaction(self, requested_heat: float, current_temp: float, ambient: float) -> float:
        heat_gradient = (current_temp - ambient) * 0.05
        return (requested_heat / self.thermal_mass) * 10.0 - heat_gradient


# ═════════════════════════════════════════════════════════════════════════════
#  3. PLANETARY NODES (Supply Chain & Planet Bloom)
# ═════════════════════════════════════════════════════════════════════════════

class PlanetaryBodyNode(Node):
    def __init__(self, planet_id: str, ambient_temp: float, is_destination=False):
        super().__init__(f'planet_{planet_id}')
        self.planet_id = planet_id
        self.ambient_temp = ambient_temp
        self.is_destination = is_destination

        self.energy = 5000.0
        self.raw_materials = 0.0
        self.finished_goods = 0.0
        self.bloom_delivered = 0.0

        self.pheromones = {"danger": 0.0, "opportunity": 0.0, "bloom_demand": 1.0}

        # Publishers/Subscribers
        self.pub_tele = self.create_publisher(Float32MultiArray, f'/{self.planet_id}/telemetry', 10,
                                              node_loc=self.planet_id)
        self.create_subscription(Float32MultiArray, f'/{self.planet_id}/agent_action', self.agent_action_cb, 10)
        self.create_subscription(Float32MultiArray, f'/global/logistics', self.logistics_cb, 10)

    def logistics_cb(self, msg):
        """Receives Interplanetary Cargo."""
        cargo_type, amount = msg.data
        if self.is_destination:
            self.bloom_delivered += amount
            self.get_logger().info(f"🌸 BLOOM RECEIVED CARGO: +{amount} Finished Goods!")
        else:
            self.finished_goods += amount

    def agent_action_cb(self, msg):
        action_type, cost, material_delta, goods_delta = msg.data
        self.energy -= cost
        self.raw_materials += material_delta
        if goods_delta > 0 and self.raw_materials >= (goods_delta * 10.0):
            self.raw_materials -= (goods_delta * 10.0)
            self.finished_goods += goods_delta

        if action_type == 4.0:  # Launch Action
            if self.finished_goods >= 1.0:
                self.finished_goods -= 1.0
                # Publish to Bloom over the Latency Bus
                latency_network.publish('/global/logistics', Float32MultiArray(data=[1.0, 1.0]), self.planet_id,
                                        "Planet_Bloom")

    def environment_tick(self):
        self.pheromones["danger"] = max(0.0, self.pheromones["danger"] - 1.0)
        self.pheromones["opportunity"] = self.raw_materials * 0.1
        # If Bloom, demand scent is high
        if self.is_destination: self.pheromones["bloom_demand"] = 10.0

        self.pub_tele.publish(Float32MultiArray(
            data=[self.energy, self.raw_materials, self.finished_goods, self.ambient_temp,
                  self.pheromones["bloom_demand"]]))

    def print_state(self):
        role = "DESTINATION" if self.is_destination else "MINE"
        print(
            f"🌍 [{self.planet_id.upper()} ({role})] Energy: {self.energy:.0f} | Mats: {self.raw_materials:.0f} | Goods: {self.finished_goods:.0f} | BLOOM TOTAL: {self.bloom_delivered:.0f}")


# ═════════════════════════════════════════════════════════════════════════════
#  4. CHASSIS MODES (Refining & Logistics)
# ═════════════════════════════════════════════════════════════════════════════

CHASSIS_CONFIGS = {
    "Hauler": {
        "emoji": "🚚",
        "modes": {
            "Active": {  # Logistics & Manufacturing
                0: {"name": "Refine Ore", "heat": 8.0, "wear": 0.1, "energy": 10.0, "goods": 1.0, "type": 1.0},
                1: {"name": "Interplanetary Launch", "heat": 25.0, "wear": 0.5, "energy": 500.0, "goods": 0.0,
                    "type": 4.0},
                2: {"name": "Thermal Vent", "heat": -15.0, "wear": -0.1, "energy": 0.0, "goods": 0.0, "type": 0.0}
            },
            "Relay_Chrysalis": {
                2: {"name": "Deep Repair", "heat": -20.0, "wear": -0.5, "energy": 0.0, "goods": 0.0, "type": 0.0},
                0: {"name": "Idle", "heat": -1.0, "wear": -0.1, "energy": 0.0, "goods": 0.0, "type": 0.0},
                1: {"name": "Idle", "heat": -1.0, "wear": -0.1, "energy": 0.0, "goods": 0.0, "type": 0.0}
            }
        }
    },
    "Borer": {
        "emoji": "🚜",
        "modes": {
            "Active": {  # Deep Mining
                0: {"name": "Deep Bore", "heat": 15.0, "wear": 0.3, "energy": 15.0, "mats": 20.0, "type": 2.0},
                1: {"name": "Surface Scrape", "heat": 4.0, "wear": 0.05, "energy": 5.0, "mats": 5.0, "type": 2.0},
                2: {"name": "Emergency Cool", "heat": -15.0, "wear": -0.1, "energy": 0.0, "mats": 0.0, "type": 0.0}
            },
            "Relay_Chrysalis": {
                2: {"name": "Deep Repair", "heat": -20.0, "wear": -0.5, "energy": 0.0, "mats": 0.0, "type": 0.0},
                0: {"name": "Idle", "heat": 0, "wear": 0, "energy": 0, "mats": 0, "type": 0},
                1: {"name": "Idle", "heat": 0, "wear": 0, "energy": 0, "mats": 0, "type": 0}}
        }
    }
}


# ═════════════════════════════════════════════════════════════════════════════
#  5. THE AGENT (Supply Chain & Expansion Aware)
# ═════════════════════════════════════════════════════════════════════════════

class StandardizedSwarmNode(Node):
    def __init__(self, name: str, chassis: str, planet_id: str, neuro_cfg: SNNNeuroConfig):
        super().__init__(f'{name}_{planet_id}')
        self.node_name, self.chassis, self.planet_id = name, chassis, planet_id
        self.config = CHASSIS_CONFIGS[chassis]
        self.physics = PhysicsConstraintSolver(chassis)
        self.software_mode = "Active"
        self.active_domain = self.config["modes"][self.software_mode]

        self.current_temp = 30.0;
        self.previous_temp = 30.0;
        self.machine_wear = 0.0
        self.last_reward, self.last_attention, self.last_mood, self.last_panic, self.last_action_taken = 0.0, 1.0, 0.0, 0.0, -1
        self.planet_tele = [0.0, 0.0, 0.0, 30.0, 1.0]  # Energy, Mats, Goods, Ambient, BloomDemand

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
        if self.machine_wear > 4.5 and self.software_mode != "Relay_Chrysalis":
            self.get_logger().warn(f"🧬 {self.node_name} Critical Wear! Metamorphosis -> Relay Chrysalis.")
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
        expected_delta = self.physics.calculate_physical_reaction(
            self.active_domain[self.last_action_taken]["heat"] if self.last_action_taken != -1 else 0.0,
            self.current_temp, self.planet_tele[3])
        mech_error_signal = np.clip((actual_temp_delta - expected_delta) / 10.0, -1.0,
                                    1.0) if self.last_action_taken != -1 else 0.0

        # Influence from Planet Bloom Demand (Index 4 of Telemetry)
        if self.planet_tele[4] > 5.0: self.last_attention += 0.8  # High demand = high attention

        proposed_action, compute_time = self.head.predict_and_learn(z=fused_emb, reward_signal=self.last_reward,
                                                                    attention_signal=self.last_attention,
                                                                    mood_signal=self.last_mood,
                                                                    mech_error_signal=mech_error_signal,
                                                                    panic_signal=self.last_panic)

        is_safe, reason = self.governor.validate_action(proposed_action, {"temperature": self.current_temp})
        self.previous_temp = self.current_temp

        if is_safe:
            self.last_mood, self.last_panic = -0.1, 0.0
            self.last_reward = 1.0 if (self.current_temp / 95.0) < 0.70 else 0.2
            self.last_action_taken = proposed_action
            action_cfg = self.active_domain[proposed_action]

            # Constraints
            physical_delta = self.physics.calculate_physical_reaction(action_cfg["heat"] + self.machine_wear,
                                                                      self.current_temp, self.planet_tele[3])
            self.current_temp += physical_delta
            self.machine_wear = max(0.0, self.machine_wear + action_cfg["wear"])

            # Resource Logic
            mats = action_cfg.get("mats", 0.0);
            goods = action_cfg.get("goods", 0.0)
            self.pub_action.publish(Float32MultiArray(data=[action_cfg["type"], action_cfg["energy"], mats, goods]),
                                    dest_loc=self.planet_id)

            self.get_logger().info(
                f"{action_cfg['name']:<20} | Temp: {self.current_temp:>4.1f}C | Wear: {self.machine_wear:>3.1f} | Bloom-Demand: {self.planet_tele[4]:.1f}")
        else:
            self.last_mood, self.last_reward, self.last_panic, self.last_action_taken = 0.5, 0.0, 1.0, -1
            self.current_temp = max(self.planet_tele[3], self.current_temp - 20.0)


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT: LOGISTICS ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def run_bloom_logistics():
    print("\n" + "═" * 80)
    print(" 🌌 PLANET FACTORY: PHASE 9 - BLOOM LOGISTICS ENGINE")
    print("═" * 80)
    b2.start_scope()

    # 1. Initialize Networked Objects
    bloom = PlanetaryBodyNode("Planet_Bloom", ambient_temp=25.0, is_destination=True)
    asteroid = PlanetaryBodyNode("Mine_A1", ambient_temp=-50.0)
    hyper_cfg = SNNNeuroConfig(base_lr=3.0)

    swarm = [
        StandardizedSwarmNode("Drill-1", "Borer", "Mine_A1", hyper_cfg),
        StandardizedSwarmNode("Refiner-1", "Hauler", "Mine_A1", hyper_cfg)
    ]

    print("\n🚀 COMMENCING BLOOM SUPPLY CHAIN...")
    sim_clock = 0.0
    for cycle in range(1, 41):
        sim_clock += 0.1
        latency_network.process_queue(sim_clock)
        bloom.environment_tick();
        asteroid.environment_tick()

        for insect in swarm: insect.cognitive_loop()

        if cycle % 10 == 0:
            print("-" * 40)
            bloom.print_state();
            asteroid.print_state()
            print("-" * 40)
        time.sleep(0.02)

    print("\n🏁 LOGISTICS SIMULATION COMPLETE.")
    bloom.print_state()
    print("═" * 80)


if __name__ == '__main__':
    run_bloom_logistics()