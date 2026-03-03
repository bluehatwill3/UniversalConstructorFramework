"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PLANET FACTORY: PHASE 3 & 4 - CYBER-PHYSICAL NODE             ║
║                                                                              ║
║  This acts as the ROS 2 Nervous System. It bridges real IoT sensors to the   ║
║  Neuromorphic Brain. Features a MODULAR Continual Learning engine integrating║
║  STDP, Reward Modulation, Homeostasis, Attention, Mood, Competition, and     ║
║  Mechanical Predictive Coding (Efference Copy / Forward Models).             ║
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
    warnings.warn("brian2 not installed. SNN Continual Learning disabled.")

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
    warnings.warn("ROS 2 (rclpy) or cv_bridge not installed. Running in standalone Offline Mode.")


    # Mock ROS 2 Classes for offline testing of the biological equations
    class Node:
        def __init__(self, name): self.name = name

        class MockLogger:
            def info(self, msg): print(f"[INFO] {msg}")

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
#  1. THE MODULAR CONTINUAL LEARNING SNN (Phase 4)
# ═════════════════════════════════════════════════════════════════════════════

class AdvancedNeuromorphicHead(nn.Module):
    """
    Stateful Spiking Neural Network with Multi-Factor Continual Learning.
    Dynamically compiles differential equations based on active learning modules.
    """

    def __init__(self, n_actions: int, learning_modules: dict = None):
        super().__init__()
        self.n_actions = n_actions

        # Modular Toggle Configuration
        self.modules = learning_modules or {
            "stdp": True,  # Spike-Timing-Dependent Plasticity
            "reward": True,  # Dopamine R-STDP
            "attention": True,  # Acetylcholine Scaling
            "homeostasis": True,  # Weight Decay Balance
            "heterosynaptic": True,  # Activity Competition
            "predictive_coding": True  # Mechanical Efference Copy / ILC
        }

        if not HAS_NEUROMORPHIC: return

        b2.start_scope()

        # Input Layer (Poisson receptors driven by PyTorch Fused Embeddings)
        self.P = b2.PoissonGroup(256, rates=np.zeros(256) * b2.Hz)

        # Output Layer (Leaky Integrate-and-Fire action neurons)
        eqs_neurons = '''
        dv/dt = (I - v) / (10*ms) : 1
        I : 1
        mood_modifier : 1  # Raises/lowers threshold based on risk perception
        '''
        self.G = b2.NeuronGroup(n_actions, eqs_neurons, threshold='v > (1.0 + mood_modifier)', reset='v=0',
                                refractory=2 * b2.ms, method='euler')

        # ── DYNAMIC BIOLOGICAL EQUATION BUILDER ──
        base_eqs = '''
        dApre/dt = -Apre / taupre : 1 (event-driven)
        dApost/dt = -Apost / taupost : 1 (event-driven)
        dc/dt = -c / tau_c : 1 (clock-driven)  
        dx/dt = (1 - x) / tau_d : 1 (clock-driven)
        '''

        # Build the dw/dt learning rule dynamically
        dw_components = []
        if self.modules["stdp"]:
            core = "c"
            if self.modules["attention"]:
                core = "(base_lr * attention) * " + core
            else:
                core = "base_lr * " + core
            if self.modules["reward"]: core += " * reward"
            dw_components.append(core)

        if self.modules["homeostasis"]:
            dw_components.append("- w_decay * w")
        if self.modules["heterosynaptic"]:
            dw_components.append("- hetero_decay * global_activity * w")
        if self.modules["predictive_coding"]:
            # Mechanical Prediction Error overrides standard learning to force physical reality alignment
            dw_components.append("- mech_lr * mech_error * c")

        dw_dt_eq = "dw/dt = " + " ".join(
            dw_components) + " : 1 (clock-driven)" if dw_components else "dw/dt = 0 : 1 (clock-driven)"

        # Build required shared variables
        shared_vars = '''
        taupre : second (shared)
        taupost : second (shared)
        tau_c : second (shared)
        tau_d : second (shared)
        base_lr : 1/second (shared)
        '''
        if self.modules["reward"]: shared_vars += "reward : 1 (shared)\n"
        if self.modules["attention"]: shared_vars += "attention : 1 (shared)\n"
        if self.modules["homeostasis"]: shared_vars += "w_decay : 1/second (shared)\n"
        if self.modules["heterosynaptic"]:
            shared_vars += "hetero_decay : 1/second (shared)\n"
            shared_vars += "global_activity : 1 (shared)\n"
        if self.modules["predictive_coding"]:
            shared_vars += "mech_error : 1 (shared)\n"
            shared_vars += "mech_lr : 1/second (shared)\n"

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
        self.S.w = 'rand() * 0.1'
        self.S.x = 1.0

        # Init biological constants
        self.S.taupre = 20 * b2.ms
        self.S.taupost = 20 * b2.ms
        self.S.tau_c = 50 * b2.ms
        self.S.tau_d = 150 * b2.ms
        self.S.base_lr = 0.5 * b2.Hz

        # Init Modular Variables safely
        if self.modules["homeostasis"]: self.S.w_decay = 0.01 * b2.Hz
        if self.modules["heterosynaptic"]:
            self.S.hetero_decay = 0.005 * b2.Hz
            self.S.global_activity = 0.0
        if self.modules["predictive_coding"]:
            self.S.mech_lr = 0.8 * b2.Hz  # Learns mechanical rules quickly
            self.S.mech_error = 0.0
        if self.modules["reward"]: self.S.reward = 0.0
        if self.modules["attention"]: self.S.attention = 1.0

        self.G.mood_modifier = 0.0
        self.M = b2.SpikeMonitor(self.G)
        self.net = b2.Network(self.P, self.G, self.S, self.M)
        self.last_counts = np.zeros(n_actions)

    def predict_and_learn(self, z: torch.Tensor, reward_signal: float, attention_signal: float, mood_signal: float,
                          mech_error_signal: float) -> int:
        """Pushes data through the SNN, applies modular multi-factor modulation, and acts."""
        if not HAS_NEUROMORPHIC: return 0

        self.P.rates = np.clip(z.squeeze().cpu().numpy(), 0, 1) * 100 * b2.Hz

        # Inject Modulators based on configuration
        if self.modules["reward"]: self.S.reward = reward_signal
        if self.modules["attention"]: self.S.attention = attention_signal
        if self.modules["predictive_coding"]: self.S.mech_error = mech_error_signal
        if self.modules["heterosynaptic"]: self.S.global_activity = float(
            np.sum(self.last_counts) / max(1, self.n_actions))
        self.G.mood_modifier = mood_signal

        # Biological state progression
        self.net.run(20 * b2.ms)

        current_counts = np.array(self.M.count)
        step_counts = current_counts - self.last_counts
        self.last_counts = current_counts.copy()

        if np.sum(step_counts) > 0: return int(np.argmax(step_counts))
        return 0

    # ═════════════════════════════════════════════════════════════════════════════


#  2. ROS 2 CYBER-PHYSICAL NODE (Phase 3)
# ═════════════════════════════════════════════════════════════════════════════

class CyberPhysicalNode(Node):
    def __init__(self):
        super().__init__('planet_factory_agent')
        self.get_logger().info('🌐 Initializing ROS 2 Cyber-Physical Nervous System (Gazebo/Webots/Hardware Bridge)...')

        if HAS_ROS2:
            self.bridge = CvBridge()
        else:
            self.bridge = None

        # ── HARDWARE SENSORS (Subscribers) ──
        self.sub_temp = self.create_subscription(Float32, '/sensor/temperature', self.temp_callback, 10)
        self.sub_img = self.create_subscription(Image, '/camera/image_raw', self.img_callback, 10)
        self.sub_lidar = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.sub_imu = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.sub_joints = self.create_subscription(JointState, '/joint_states', self.joints_callback, 10)
        self.sub_force = self.create_subscription(WrenchStamped, '/force_torque_sensor', self.force_callback, 10)

        # ── HARDWARE ACTUATORS (Publishers) ──
        self.pub_base = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_arm = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)

        # Internal Real-Time State Buffers
        self.current_temp = 40.0
        self.previous_temp = 40.0
        self.current_img = np.zeros((3, 224, 224), dtype=np.float32)
        self.current_joints = np.zeros((32, 3), dtype=np.float32)

        # Advanced Modulator Buffers
        self.last_reward = 0.0
        self.last_attention = 1.0
        self.last_mood = 0.0
        self.last_action_taken = -1  # Used for Predictive Coding expectations

        # Load AI Architectures
        self.head = AdvancedNeuromorphicHead(n_actions=3)
        self.governor = SafetyGovernor(IndustrialThresholds(max_temp=85.0))

        self.norm = FeatureNormalizer()
        self.perception = PerceptionPipeline(self.norm, HoloSynHeads(), StudentDistilledHeadsHF(),
                                             StudentDistilledHeadsBasic())

        if HAS_ROS2: self.timer = self.create_timer(0.1, self.cognitive_loop)
        self.get_logger().info('🧠 Multi-Factor Continual Learning SNN Online.')

    # ── Sensor Callbacks ──
    def temp_callback(self, msg):
        self.current_temp = msg.data

    def img_callback(self, msg):
        if not HAS_ROS2 or self.bridge is None or not HAS_CV2: return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv_image_resized = cv2.resize(cv_image, (224, 224))
            self.current_img = cv_image_resized[:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.0
        except Exception as e:
            self.get_logger().warn(f"Image processing failed: {e}")

    def lidar_callback(self, msg):
        if not HAS_ROS2: return
        ranges = np.nan_to_num(np.array(msg.ranges), posinf=10.0, neginf=0.0)
        self.current_joints[14] = [np.mean(ranges), np.min(ranges), np.max(ranges)]

    def imu_callback(self, msg):
        if not HAS_ROS2: return
        self.current_joints[10] = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        self.current_joints[11] = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]

    def force_callback(self, msg):
        if not HAS_ROS2: return
        self.current_joints[12] = [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]
        self.current_joints[13] = [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]

    def joints_callback(self, msg):
        if not HAS_ROS2: return
        for i, pos in enumerate(msg.position[:min(32, len(msg.position))]):
            self.current_joints[i, 0] = pos

    def cognitive_loop(self):
        """The core tick of the Cyber-Physical brain."""
        bundle = ModalityBundle(
            E_t_aug=torch.randn(1, 528),
            E_i=torch.from_numpy(self.current_img).unsqueeze(0).flatten(1)[:, :2048],
            E_a=torch.randn(1, 2048), E_v=torch.randn(1, 2048),
            H=torch.from_numpy(self.current_joints).unsqueeze(0),
            feat_hf=torch.randn(1, 789), feat_basic=torch.randn(1, 5), mask=torch.ones(1, 4),
        )
        fused_emb, _, _ = self.perception.perceive(bundle)

        # ── 1. PREDICTIVE CODING: Forward Model Expectation Error ──
        # The AI compares the physical reality to what it expected to happen based on its last action
        actual_temp_delta = self.current_temp - self.previous_temp

        expected_delta = 0.0
        if self.last_action_taken == 0:
            expected_delta = 5.0  # Expected base movement to cause 5C heat
        elif self.last_action_taken == 1:
            expected_delta = 5.0  # Expected arm movement to cause 5C heat
        elif self.last_action_taken == -1:
            expected_delta = -15.0  # Expected forced stop to cool 15C

        mech_error_raw = actual_temp_delta - expected_delta
        mech_error_signal = np.clip(mech_error_raw / 10.0, -1.0, 1.0)  # Normalized mechanical error

        if abs(mech_error_signal) > 0.2:
            self.get_logger().info(
                f"⚙️ Mech/Prediction Error detected! Expected {expected_delta}C delta, got {actual_temp_delta:.1f}C. Correcting forward model.")

        # ── 2. SNN INFERENCE & PLASTICITY UPDATE ──
        proposed_action = self.head.predict_and_learn(
            z=fused_emb,
            reward_signal=self.last_reward,
            attention_signal=self.last_attention,
            mood_signal=self.last_mood,
            mech_error_signal=mech_error_signal
        )

        # ── 3. SAFETY & ENVIRONMENTAL PROCESSING ──
        metrics = {"temperature": self.current_temp}
        is_safe, reason = self.governor.validate_action(proposed_action, metrics)
        self.previous_temp = self.current_temp
        temp_ratio = self.current_temp / 85.0

        # Attention spikes during physical state-changes
        self.last_attention = 1.0 + min(2.0, abs(actual_temp_delta) * 0.5)

        # ── 4. ACTUATION & MODULATOR PREPARATION ──
        if is_safe:
            self.last_mood = -0.1  # Relax threshold

            if temp_ratio < 0.75:
                self.last_reward = 1.0
            else:
                self.last_reward = 0.2
                self.get_logger().info(f"⚠️ Temp Elevated ({self.current_temp:.1f}C). Dopamine throttled.")

            if actual_temp_delta < 0:
                self.last_reward += 0.5
                self.get_logger().info("❄️ Relief Bonus: System cooled.")

            # Hardware Actuation
            self.last_action_taken = proposed_action
            if proposed_action == 0:
                msg = Twist()
                msg.linear.x = 0.5
                self.pub_base.publish(msg)
            elif proposed_action == 1:
                if HAS_ROS2:
                    traj_msg = JointTrajectory()
                    traj_msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint']
                    point = JointTrajectoryPoint()
                    point.positions = [0.5, -0.5]
                    traj_msg.points.append(point)
                    self.pub_arm.publish(traj_msg)

            self.get_logger().info(
                f"✅ Safe Action: {proposed_action} | Attn: {self.last_attention:.2f} | Mech Error: {mech_error_signal:.2f}")
            if not HAS_ROS2: self.current_temp += 5.0
        else:
            self.last_mood = +0.5  # Raise threshold (Risk Averse)
            self.last_reward = 0.0  # Neutralize dopamine (Homeostatic override)
            self.last_action_taken = -1  # Flag for forced stop

            self.get_logger().warn(f"🛑 BLOCKED ({reason}): Action aborted.")

            self.pub_base.publish(Twist())
            if HAS_ROS2: self.pub_arm.publish(JointTrajectory())
            if not HAS_ROS2: self.current_temp = max(40.0, self.current_temp - 15.0)


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main(args=None):
    if HAS_ROS2:
        rclpy.init(args=args)
        node = CyberPhysicalNode()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    else:
        print("\n⚠️ ROS 2 not detected. Running Biological STDP Simulation offline...\n")
        node = CyberPhysicalNode()
        for i in range(20):
            node.cognitive_loop()
            time.sleep(0.1)


if __name__ == '__main__':
    main()