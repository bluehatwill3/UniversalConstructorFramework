"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PLANET FACTORY: PHASE 3 & 4 - CYBER-PHYSICAL NODE             ║
║                                                                              ║
║  This acts as the ROS 2 Nervous System. It bridges real IoT sensors to the   ║
║  Neuromorphic Brain. It features Continual Learning via STDP, Reward         ║
║  Modulation (R-STDP), and Homeostasis.                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import warnings

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
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float32, String
    from geometry_msgs.msg import Twist
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    warnings.warn("ROS 2 (rclpy) not installed. Running in standalone Offline Mode.")

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
    class Float32: pass
    class String: pass
    class Image: pass


# ═════════════════════════════════════════════════════════════════════════════
#  1. THE CONTINUAL LEARNING SNN (Phase 4)
# ═════════════════════════════════════════════════════════════════════════════

class AdvancedNeuromorphicHead(nn.Module):
    """
    Stateful Spiking Neural Network with Continual Learning.
    Integrates STDP, Dopamine Reward Traces, and Homeostasis.
    """
    def __init__(self, n_actions: int):
        super().__init__()
        self.n_actions = n_actions
        if not HAS_NEUROMORPHIC: return

        b2.start_scope()

        # Input Layer (Poisson receptors driven by PyTorch Fused Embeddings)
        self.P = b2.PoissonGroup(256, rates=np.zeros(256)*b2.Hz)

        # Output Layer (Leaky Integrate-and-Fire action neurons)
        eqs_neurons = '''
        dv/dt = (I - v) / (10*ms) : 1
        I : 1
        '''
        self.G = b2.NeuronGroup(n_actions, eqs_neurons, threshold='v>1', reset='v=0', refractory=2*b2.ms, method='euler')

        # ── THE BIOLOGICAL LEARNING EQUATIONS ──
        syn_eqs = '''
        dApre/dt = -Apre / taupre : 1 (event-driven)
        dApost/dt = -Apost / taupost : 1 (event-driven)
        
        # Eligibility trace (c) stores the STDP overlap history
        dc/dt = -c / tau_c : 1 (clock-driven)  
        
        # Short-Term Depression recovery (available neurotransmitter resources)
        dx/dt = (1 - x) / tau_d : 1 (clock-driven)
        
        # The Master Learning Rule: (STDP * Reward) - (Homeostatic Decay)
        dw/dt = learning_rate * c * reward - w_decay * w : 1 (clock-driven)
        
        reward : 1 (shared)
        taupre : second (shared)
        taupost : second (shared)
        tau_c : second (shared)
        tau_d : second (shared)
        learning_rate : 1/second (shared)
        w_decay : 1/second (shared)
        '''

        self.S = b2.Synapses(self.P, self.G, model=syn_eqs,
                             on_pre='''v_post += w * x
                                       x *= 0.8  # Deplete resources on spike (STP)
                                       Apre += 0.01
                                       c += Apost''',
                             on_post='''Apost -= 0.01
                                        c += Apre''',
                             method='euler')

        self.S.connect()
        self.S.w = 'rand() * 0.1' # Initialize small random weights
        self.S.x = 1.0 # Initialize STP resources to max

        # Biological constants
        self.S.taupre = 20*b2.ms
        self.S.taupost = 20*b2.ms
        self.S.tau_c = 50*b2.ms
        self.S.tau_d = 150*b2.ms  # STP resource recovery time
        self.S.learning_rate = 0.5 * b2.Hz
        self.S.w_decay = 0.01 * b2.Hz
        self.S.reward = 0.0 # Will be injected dynamically by the Safety Governor

        self.M = b2.SpikeMonitor(self.G)
        self.net = b2.Network(self.P, self.G, self.S, self.M)
        self.last_counts = np.zeros(n_actions)

    def predict_and_learn(self, z: torch.Tensor, reward_signal: float) -> int:
        """Pushes data through the SNN, updates weights based on dopamine/reward, and acts."""
        if not HAS_NEUROMORPHIC: return 0

        # Drive the input receptors
        snn_input_rates = np.clip(z.squeeze().cpu().numpy(), 0, 1) * 100 * b2.Hz
        self.P.rates = snn_input_rates

        # Inject Dopamine/Reward signal (Modulates the STDP Eligibility Trace)
        self.S.reward = reward_signal

        # Run biological simulation for 20ms (Maintains continuous state)
        self.net.run(20*b2.ms)

        # Calculate spikes that occurred in this exact 20ms window
        current_counts = np.array(self.M.count)
        step_counts = current_counts - self.last_counts
        self.last_counts = current_counts.copy()

        if np.sum(step_counts) > 0:
            return int(np.argmax(step_counts))
        return 0


# ═════════════════════════════════════════════════════════════════════════════
#  2. ROS 2 CYBER-PHYSICAL NODE (Phase 3)
# ═════════════════════════════════════════════════════════════════════════════

class CyberPhysicalNode(Node):
    def __init__(self):
        super().__init__('planet_factory_agent')
        self.get_logger().info('🌐 Initializing ROS 2 Cyber-Physical Nervous System...')

        # ── HARDWARE SENSORS (Subscribers) ──
        self.sub_temp = self.create_subscription(Float32, '/sensor/temperature', self.temp_callback, 10)
        self.sub_img = self.create_subscription(Image, '/camera/image_raw', self.img_callback, 10)

        # ── HARDWARE ACTUATORS (Publishers) ──
        self.pub_action = self.create_publisher(Twist, '/cmd_vel', 10)

        # Internal Real-Time State
        self.current_temp = 40.0
        self.current_img = np.zeros((3, 224, 224), dtype=np.float32)
        self.last_reward = 0.0 # Tracks delayed dopamine signal

        # Load AI Architectures
        self.head = AdvancedNeuromorphicHead(n_actions=3)
        self.governor = SafetyGovernor(IndustrialThresholds(max_temp=85.0))

        self.norm = FeatureNormalizer()
        self.perception = PerceptionPipeline(self.norm, HoloSynHeads(), StudentDistilledHeadsHF(), StudentDistilledHeadsBasic())

        # Cognitive Loop (Runs at 10 Hz)
        if HAS_ROS2:
            self.timer = self.create_timer(0.1, self.cognitive_loop)
        self.get_logger().info('🧠 Continual Learning SNN Online.')

    def temp_callback(self, msg):
        """IoT Thermocouple stream updates real-time temperature."""
        self.current_temp = msg.data

    def img_callback(self, msg):
        """IoT Camera stream (Requires cv_bridge in a full ROS2 stack)."""
        pass

    def cognitive_loop(self):
        """The core tick of the Cyber-Physical brain."""
        # 1. Package Modalities
        bundle = ModalityBundle(
            E_t_aug=torch.randn(1, 528),
            E_i=torch.from_numpy(self.current_img).unsqueeze(0).flatten(1)[:, :2048],
            E_a=torch.randn(1, 2048), E_v=torch.randn(1, 2048),
            H=torch.zeros(1, 32, 3), feat_hf=torch.randn(1, 789),
            feat_basic=torch.randn(1, 5), mask=torch.ones(1, 4),
        )

        # 2. Forward pass through frozen PyTorch pipeline
        fused_emb, _, _ = self.perception.perceive(bundle)

        # 3. SNN Inference & Plasticity Update
        # Note: We pass the reward from the *previous* cycle. Biological dopamine is delayed!
        proposed_action = self.head.predict_and_learn(fused_emb, self.last_reward)

        # 4. Amygdala Safety Check
        metrics = {"temperature": self.current_temp}
        is_safe, reason = self.governor.validate_action(proposed_action, metrics)

        # 5. Actuate & Generate Next Dopamine Reward
        msg = Twist()
        if is_safe:
            self.last_reward = 1.0  # Reinforce this synaptic pathway

            # Translate to physical ROS 2 Velocity Command
            if proposed_action == 0: msg.linear.x = 0.5
            elif proposed_action == 1: msg.angular.z = 0.5

            self.pub_action.publish(msg)
            self.get_logger().info(f"✅ Safe Action Executed: {proposed_action}")

            # Simulate physical heat buildup for offline testing
            if not HAS_ROS2: self.current_temp += 5.0
        else:
            self.last_reward = -2.0  # Strongly penalize synapses that caused this block via R-STDP
            self.get_logger().warn(f"🛑 BLOCKED ({reason}): SNN synapses heavily penalized.")
            self.pub_action.publish(Twist()) # Publish zero velocity

            # Simulate cooling down
            if not HAS_ROS2: self.current_temp = max(40.0, self.current_temp - 15.0)

# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main(args=None):
    if HAS_ROS2:
        # Standard ROS 2 Execution
        rclpy.init(args=args)
        node = CyberPhysicalNode()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    else:
        # Standalone Biological Learning Simulation
        print("\n⚠️ ROS 2 not detected. Running Biological STDP Simulation offline...\n")
        node = CyberPhysicalNode()
        for i in range(20): # Run 20 ticks
            node.cognitive_loop()
            time.sleep(0.1)

if __name__ == '__main__':
    main()