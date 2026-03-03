"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PLANET FACTORY: PHASE 12 - CYBER-ECONOMIC CONSCIOUSNESS       ║
║                                                                              ║
║  Agents now possess Economic Senses. The SNN literally "feels" resource      ║
║  scarcity and experiences biological panic (Noradrenaline) if it attempts    ║
║  actions it cannot afford. Features Thermoelectric Grid Recovery and         ║
║  deadlock-free Supply Chain logic.                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import warnings
import math
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


# ── 🌐 ASYNCHRONOUS INTERPLANETARY LOGISTICS BUS ─────────────────────────────
@dataclass
class NetworkPacket:
    topic: str
    payload: Any
    deliver_at: float
    source: str
    dest: str


class LatencyEventBus:
    """Models speed-of-light comms and orbital transfer windows."""

    def __init__(self):
        self.topics = defaultdict(list)
        self.queue = deque()
        self.current_sim_time = 0.0

    def subscribe(self, topic, callback):
        self.topics[topic].append(callback)

    def publish(self, topic, msg, source_loc: str, dest_loc: str):
        latency = 0.01
        if source_loc != dest_loc and "GLOBAL" not in [source_loc, dest_loc]:
            # Rocket Transport Latency (Simulated Orbital Transfer Time)
            latency = 10.0 if dest_loc == "Planet_Bloom" else 15.0

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
#  1. UNRESTRICTED SNN
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
#  2. PHYSICS & KINEMATICS
# ═════════════════════════════════════════════════════════════════════════════

class PhysicsConstraintSolver:
    def __init__(self, chassis: str):
        mass_map = {"Borer": 50.0, "Foundry": 80.0, "Assembler": 40.0, "Synthesizer": 60.0, "Orbital_Lander": 150.0}
        self.thermal_mass = mass_map.get(chassis, 30.0)

    def calculate_physical_reaction(self, requested_heat: float, current_temp: float, ambient: float) -> float:
        heat_gradient = (current_temp - ambient) * 0.08
        return (requested_heat / self.thermal_mass) * 15.0 - heat_gradient


# ═════════════════════════════════════════════════════════════════════════════
#  3. LIVING PLANET & DEEP ECONOMY MODEL
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PlanetEconomy:
    energy: float = 10000.0
    ore: float = 500.0
    semi_finished: float = 0.0
    finished_goods: float = 0.0
    propellant: float = 0.0
    infrastructure: float = 1.0


class LivingPlanetNode(Node):
    """Simulates a dynamic planet with weather, day/night cycles, and a deep economy."""

    def __init__(self, planet_id: str, base_temp: float, is_destination=False):
        super().__init__(f'planet_{planet_id}')
        self.planet_id = planet_id
        self.base_temp = base_temp
        self.current_ambient = base_temp
        self.is_destination = is_destination

        self.time_of_day = 0.0  # 0 to 24
        self.economy = PlanetEconomy()
        self.bloom_delivered = 0.0
        self.pheromones = {"danger": 0.0, "opportunity": 0.0, "bloom_demand": 1.0}

        self.pub_tele = self.create_publisher(Float32MultiArray, f'/{self.planet_id}/telemetry', 10,
                                              node_loc=self.planet_id)
        self.create_subscription(Float32MultiArray, f'/{self.planet_id}/agent_action', self.agent_action_cb, 10)
        self.create_subscription(Float32MultiArray, f'/global/logistics', self.logistics_cb, 10)

    def logistics_cb(self, msg):
        cargo_type, amount = msg.data
        if self.is_destination:
            self.bloom_delivered += amount
            self.get_logger().info(f"🌸 BLOOM RECEIVED CARGO: +{amount} Finished Goods from orbit!")
        else:
            self.economy.finished_goods += amount

    def agent_action_cb(self, msg):
        act_class, e_in, ore_in, semi_in, goods_in, prop_in, ore_out, semi_out, goods_out, prop_out = msg.data

        # We subtract energy safely, allowing robots to recharge the grid (negative e_in adds energy)
        self.economy.energy -= e_in
        self.economy.ore = max(0, self.economy.ore - ore_in + ore_out)
        self.economy.semi_finished = max(0, self.economy.semi_finished - semi_in + semi_out)
        self.economy.finished_goods = max(0, self.economy.finished_goods - goods_in + goods_out)
        self.economy.propellant = max(0, self.economy.propellant - prop_in + prop_out)

        if act_class == 9.0 and goods_in > 0:
            latency_network.publish('/global/logistics', Float32MultiArray(data=[1.0, goods_in]), self.planet_id,
                                    "Planet_Bloom")
            self.get_logger().info(f"🚀 ROCKET LAUNCH DETECTED: Sending {goods_in} Goods to Orbit!")

    def environment_tick(self):
        self.time_of_day = (self.time_of_day + 0.5) % 24.0
        solar_intensity = math.sin(self.time_of_day * (math.pi / 12.0))
        self.current_ambient = self.base_temp + (solar_intensity * 30.0)

        # Buffed Solar Output to sustain macro-economic loads
        if solar_intensity > 0:
            self.economy.energy += (solar_intensity * 200.0 * self.economy.infrastructure)

        self.pheromones["danger"] = max(0.0, self.pheromones["danger"] - 1.0)
        if self.is_destination: self.pheromones["bloom_demand"] = 10.0

        tele_data = [self.economy.energy, self.economy.ore, self.economy.semi_finished,
                     self.economy.finished_goods, self.economy.propellant,
                     self.current_ambient, self.pheromones["bloom_demand"]]
        self.pub_tele.publish(Float32MultiArray(data=tele_data))

    def print_state(self):
        role = "DESTINATION" if self.is_destination else "LIVING COLONY"
        print(
            f"🌍 [{self.planet_id.upper()} ({role})] Time: {self.time_of_day:02.0f}:00 | Ambient: {self.current_ambient:5.1f}C")
        print(
            f"   💰 Economy | NRG: {self.economy.energy:5.0f} | ORE: {self.economy.ore:4.0f} | SEMI: {self.economy.semi_finished:3.0f} | GDS: {self.economy.finished_goods:3.0f} | FUEL: {self.economy.propellant:4.0f}")
        if self.is_destination: print(f"   🌸 BLOOM TOTAL: {self.bloom_delivered:.0f}")


# ═════════════════════════════════════════════════════════════════════════════
#  4. DEEP ECONOMIC CHASSIS MODES (Recipes Optimized for Energy Recovery)
# ═════════════════════════════════════════════════════════════════════════════

def recipe(cls, e, oi, si, gi, pi, oo, so, go, po):
    return [float(x) for x in [cls, e, oi, si, gi, pi, oo, so, go, po]]


CHASSIS_CONFIGS = {
    "Borer": {
        "emoji": "🚜",
        "modes": {
            "Active": {
                0: {"name": "Deep Mine Ore", "heat": 15.0, "wear": 0.2,
                    "recipe": recipe(1, 20, 0, 0, 0, 0, 40, 0, 0, 0)},
                1: {"name": "Surface Scrape", "heat": 5.0, "wear": 0.05,
                    "recipe": recipe(1, 5, 0, 0, 0, 0, 10, 0, 0, 0)},
                # Action 2 Recharges the grid (Thermoelectric/Solar)
                2: {"name": "Rest/Cool", "heat": -15.0, "wear": -0.1, "recipe": recipe(0, -15, 0, 0, 0, 0, 0, 0, 0, 0)}
            }
        }
    },
    "Foundry": {
        "emoji": "🔥",
        "modes": {
            "Active": {
                0: {"name": "Smelt Alloys", "heat": 40.0, "wear": 0.3,
                    "recipe": recipe(2, 50, 20, 0, 0, 0, 0, 10, 0, 0)},
                1: {"name": "Purify Silicon", "heat": 20.0, "wear": 0.1,
                    "recipe": recipe(2, 30, 10, 0, 0, 0, 0, 5, 0, 0)},
                2: {"name": "Vent Furnaces", "heat": -30.0, "wear": -0.1,
                    "recipe": recipe(0, -20, 0, 0, 0, 0, 0, 0, 0, 0)}
            }
        }
    },
    "Assembler": {
        "emoji": "🦾",
        "modes": {
            "Active": {
                0: {"name": "Heavy Assembly", "heat": 15.0, "wear": 0.1,
                    "recipe": recipe(3, 40, 0, 10, 0, 0, 0, 0, 2, 0)},
                1: {"name": "Precision Fab", "heat": 5.0, "wear": 0.02,
                    "recipe": recipe(3, 15, 0, 5, 0, 0, 0, 0, 1, 0)},
                2: {"name": "Diagnostic", "heat": -10.0, "wear": -0.1, "recipe": recipe(0, -10, 0, 0, 0, 0, 0, 0, 0, 0)}
            }
        }
    },
    "Synthesizer": {
        "emoji": "⚗️",
        "modes": {
            "Active": {
                0: {"name": "Crack Propellant", "heat": 25.0, "wear": 0.2,
                    "recipe": recipe(4, 100, 5, 0, 0, 0, 0, 0, 0, 50)},
                1: {"name": "Mix Oxidizer", "heat": 10.0, "wear": 0.05,
                    "recipe": recipe(4, 40, 2, 0, 0, 0, 0, 0, 0, 15)},
                2: {"name": "Flush Pipes", "heat": -20.0, "wear": -0.1,
                    "recipe": recipe(0, -15, 0, 0, 0, 0, 0, 0, 0, 0)}
            }
        }
    },
    "Orbital_Lander": {
        "emoji": "🚀",
        "modes": {
            "Active": {
                0: {"name": "INTERPLANETARY LAUNCH", "heat": 150.0, "wear": 1.5,
                    "recipe": recipe(9, 500, 0, 0, 2, 100, 0, 0, 0, 0)},
                1: {"name": "Sub-Orbital Hop", "heat": 50.0, "wear": 0.5,
                    "recipe": recipe(8, 200, 0, 0, 0, 40, 0, 0, 0, 0)},
                2: {"name": "Cryo-Chill", "heat": -60.0, "wear": -0.2, "recipe": recipe(0, -50, 0, 0, 0, 0, 0, 0, 0, 0)}
            }
        }
    }
}


# ═════════════════════════════════════════════════════════════════════════════
#  5. THE AGENT (SNN linked to Deep Economy & Scarcity)
# ═════════════════════════════════════════════════════════════════════════════

class StandardizedSwarmNode(Node):
    def __init__(self, name: str, chassis: str, planet_id: str, neuro_cfg: SNNNeuroConfig):
        super().__init__(f'{name}_{planet_id}')
        self.node_name, self.chassis, self.planet_id = name, chassis, planet_id
        self.config = CHASSIS_CONFIGS[chassis]
        self.physics = PhysicsConstraintSolver(chassis)
        self.active_domain = self.config["modes"]["Active"]

        self.current_temp, self.previous_temp, self.machine_wear = 30.0, 30.0, 0.0
        self.last_reward, self.last_attention, self.last_mood, self.last_panic, self.last_action_taken = 0.0, 1.0, 0.0, 0.0, -1
        # [Energy, Ore, Semi, Goods, Prop, Ambient, BloomDemand]
        self.planet_tele = [10000.0, 500.0, 0.0, 0.0, 0.0, 30.0, 1.0]

        max_t = 200.0 if chassis == "Orbital_Lander" else 100.0
        self.governor = SafetyGovernor(IndustrialThresholds(max_temp=max_t))
        self.head = AdvancedNeuromorphicHead(n_actions=3, neuro_config=neuro_cfg)
        self.perception = PerceptionPipeline(FeatureNormalizer(), HoloSynHeads(), StudentDistilledHeadsHF(),
                                             StudentDistilledHeadsBasic())

        self.pub_action = self.create_publisher(Float32MultiArray, f'/{self.planet_id}/agent_action', 10,
                                                node_loc=self.planet_id)
        self.create_subscription(Float32MultiArray, f'/{self.planet_id}/telemetry', self.telemetry_cb, 10)

    def telemetry_cb(self, msg):
        self.planet_tele = msg.data

    def cognitive_loop(self):
        # ── ECONOMIC SENSORY INJECTION ──
        # The SNN literally feels the wealth/poverty of the planet via normalized senses (0.0 to 1.0)
        econ_senses = torch.tensor([[
            min(1.0, max(0.0, self.planet_tele[0] / 10000.0)),  # Energy
            min(1.0, max(0.0, self.planet_tele[1] / 1000.0)),  # Ore
            min(1.0, max(0.0, self.planet_tele[2] / 100.0)),  # Semi-Finished
            min(1.0, max(0.0, self.planet_tele[3] / 100.0)),  # Goods
            min(1.0, max(0.0, self.planet_tele[4] / 500.0))  # Propellant
        ]], dtype=torch.float32)

        bundle = ModalityBundle(
            E_t_aug=torch.randn(1, 528), E_i=torch.zeros(1, 2048), E_a=torch.randn(1, 2048), E_v=torch.randn(1, 2048),
            H=torch.zeros(1, 32, 3), feat_hf=torch.randn(1, 789),
            feat_basic=econ_senses,  # Integrated Economic Consciousness!
            mask=torch.ones(1, 4)
        )
        fused_emb, _, _ = self.perception.perceive(bundle)

        ambient_t = self.planet_tele[5]

        if self.planet_tele[6] > 5.0: self.last_attention += 0.5

        proposed_action, compute_time = self.head.predict_and_learn(z=fused_emb, reward_signal=self.last_reward,
                                                                    attention_signal=self.last_attention,
                                                                    mood_signal=self.last_mood, mech_error_signal=0.0,
                                                                    panic_signal=self.last_panic)

        # ── THE ECONOMIC ULTIMATUM (Deadlock Fix applied) ──
        action_cfg = self.active_domain[proposed_action]
        rec = action_cfg.get("recipe", [0] * 10)

        # Check if we can afford the action (allowing negative or zero energy actions to bypass low energy deadlocks)
        can_afford = ((self.planet_tele[0] >= rec[1] or rec[1] <= 0) and
                      self.planet_tele[1] >= rec[2] and
                      self.planet_tele[2] >= rec[3] and
                      self.planet_tele[3] >= rec[4] and
                      self.planet_tele[4] >= rec[5])

        is_safe, thermal_reason = self.governor.validate_action(proposed_action, {"temperature": self.current_temp})
        self.previous_temp = self.current_temp

        reason = thermal_reason if not is_safe else ("SCARCITY" if not can_afford else "OK")
        is_valid = is_safe and can_afford

        if is_valid:
            self.last_mood, self.last_panic = -0.1, 0.0

            thermal_ratio = self.current_temp / self.governor.limits.max_temp
            self.last_reward = 1.5 if thermal_ratio < 0.60 else 0.3

            # ⚡ Dynamic Grid Recovery Bonus
            if rec[1] < 0 and self.planet_tele[0] < 2000.0:
                self.last_reward += 1.0
                self.get_logger().info(f"⚡ {self.node_name} Grid Recovery Bonus applied!")

            self.last_action_taken = proposed_action

            physical_delta = self.physics.calculate_physical_reaction(action_cfg["heat"] + self.machine_wear,
                                                                      self.current_temp, ambient_t)
            self.current_temp += physical_delta
            self.machine_wear = max(0.0, self.machine_wear + action_cfg["wear"])

            self.pub_action.publish(Float32MultiArray(data=action_cfg["recipe"]), dest_loc=self.planet_id)

            self.get_logger().info(
                f"{self.config['emoji']} {self.node_name:<10} | {action_cfg['name']:<22} | Temp: {self.current_temp:>5.1f}C (Env: {ambient_t:4.0f}C)")
        else:
            self.last_mood, self.last_reward, self.last_panic, self.last_action_taken = 0.8, 0.0, 1.0, -1
            self.current_temp = max(ambient_t, self.current_temp - 30.0)
            self.get_logger().warn(
                f"{self.config['emoji']} {self.node_name:<10} | BLOCKED ({reason})! Noradrenaline injected.")


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT: LIVING PLANET SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

def run_living_planet_economy():
    print("\n" + "═" * 90)
    print(" 🌌 PLANET FACTORY: PHASE 12 - CYBER-ECONOMIC CONSCIOUSNESS")
    print("═" * 90)
    b2.start_scope()

    # 1. Initialize Living Environments
    bloom = LivingPlanetNode("Planet_Bloom", base_temp=20.0, is_destination=True)
    mine_world = LivingPlanetNode("Vulcan_Prime", base_temp=50.0)

    hyper_cfg = SNNNeuroConfig(base_lr=3.5)

    # 2. Instantiate a complete Macro-Economic Swarm
    swarm = [
        StandardizedSwarmNode("Borer-Alpha", "Borer", "Vulcan_Prime", hyper_cfg),
        StandardizedSwarmNode("Forge-1", "Foundry", "Vulcan_Prime", hyper_cfg),
        StandardizedSwarmNode("Fab-Unit", "Assembler", "Vulcan_Prime", hyper_cfg),
        StandardizedSwarmNode("Chem-Plant", "Synthesizer", "Vulcan_Prime", hyper_cfg),
        StandardizedSwarmNode("Apollo-1", "Orbital_Lander", "Vulcan_Prime", hyper_cfg)
    ]

    print("\n🚀 COMMENCING MACRO-ECONOMIC SIMULATION...")
    sim_clock = 0.0
    for cycle in range(1, 71):
        sim_clock += 0.5
        latency_network.process_queue(sim_clock)

        bloom.environment_tick()
        mine_world.environment_tick()

        for insect in swarm: insect.cognitive_loop()

        if cycle % 10 == 0:
            print("-" * 60)
            bloom.print_state()
            mine_world.print_state()
            print("-" * 60)
        time.sleep(0.01)

    print("\n🏁 INTERPLANETARY ECONOMY SIMULATION END.")
    bloom.print_state()
    print("═" * 90)


if __name__ == '__main__':
    run_living_planet_economy()