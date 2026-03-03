"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PLANET FACTORY: PHASE 2 - DIGITAL TWIN EMBODIMENT             ║
║                                                                              ║
║  This engine bridges the PyBullet 3D physics simulator with our UCF v3.0     ║
║  architecture. It captures live synthetic camera feeds, passes them to the   ║
║  Neuromorphic Action Head, and translates the SNN output into joint torques. ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import torch
import os

# Import our Brains and Architecture from Phase 1
from ucf import (
    IntelligentMachine, PerceptionPipeline, FeatureNormalizer,
    HoloSynHeads, StudentDistilledHeadsHF, StudentDistilledHeadsBasic,
    ModalityBundle, NORM_JSON, DEVICE, SafetyGovernor, IndustrialThresholds
)

# ═════════════════════════════════════════════════════════════════════════════
#  1. PHYSICS ENVIRONMENT SETUP
# ═════════════════════════════════════════════════════════════════════════════

class FactoryDigitalTwin:
    def __init__(self):
        print("🌍 Initializing PyBullet Digital Twin...")
        # Start the physics engine with a GUI so you can watch it
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load the factory floor and a KUKA robotic arm
        self.planeId = p.loadURDF("plane.urdf")

        # Load the arm and get its joint count
        self.robotId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robotId)

        # Camera setup for the robot's "eyes"
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0.5], distance=1.5,
            yaw=45, pitch=-30, roll=0, upAxisIndex=2
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
        )

        print(f"🦾 KUKA Arm spawned with {self.num_joints} joints.")

        # Add visual overlay for AI thoughts
        self.ai_text_id = p.addUserDebugText("Booting SNN Brain...", [0, 0, 1.2], textColorRGB=[1, 1, 0], textSize=2)
        self.simulated_temp = 40.0 # Base physical temperature

    def get_robot_camera_feed(self) -> np.ndarray:
        """Captures a 224x224 RGB image from the physics engine."""
        width, height, rgb_pixels, _, _ = p.getCameraImage(
            224, 224, self.view_matrix, self.proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        # Reshape and normalize for PyTorch vision encoders (C, H, W)
        img = np.reshape(rgb_pixels, (height, width, 4))[:, :, :3] # Drop Alpha channel
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        return img

    def get_joint_states(self) -> np.ndarray:
        """Extracts proprioceptive haptic data (joint positions)."""
        states = p.getJointStates(self.robotId, range(self.num_joints))
        positions = [state[0] for state in states]
        # Pad or slice to match our 32-dim haptic tensor requirement
        haptic_array = np.zeros((32, 3), dtype=np.float32)
        for i, pos in enumerate(positions[:min(32, len(positions))]):
            haptic_array[i, 0] = pos # Just mapping position to the first dimension
        return haptic_array

    def apply_action(self, action_id: int):
        """Translates discrete SNN output to physical motor torques."""
        # Action Map for 'robot_arm': 0: Move, 1: Engage
        target_vel = 0.0
        if action_id == 0:
            target_vel = 0.5  # Sweep base
            self.simulated_temp += 0.2 # Moving generates heat
        elif action_id == 1:
            target_vel = -0.5 # Sweep opposite way
            self.simulated_temp += 0.5 # Engaging generates more heat
        else:
            self.simulated_temp = max(40.0, self.simulated_temp - 1.0) # Cool down if blocked/resting

        # Apply velocity to the base swivel joint (Joint 0)
        p.setJointMotorControl2(
            bodyUniqueId=self.robotId, jointIndex=0,
            controlMode=p.VELOCITY_CONTROL, targetVelocity=target_vel, force=50
        )
        # Step the physics engine forward 1/240th of a second
        p.stepSimulation()

    def update_hud(self, action_name: str, is_safe: bool, temp: float):
        """Updates the floating text above the robot to prove AI control."""
        color = [0, 1, 0] if is_safe else [1, 0, 0] # Green if Safe, Red if Blocked
        status = f"AI: {action_name} | Temp: {temp:.1f}C"
        self.ai_text_id = p.addUserDebugText(
            status, [0, 0, 1.2], textColorRGB=color, textSize=1.5,
            replaceItemUniqueId=self.ai_text_id
        )

# ═════════════════════════════════════════════════════════════════════════════
#  2. EMBODIED SIMULATION LOOP
# ═════════════════════════════════════════════════════════════════════════════

def run_embodied_simulation():
    print("═" * 70)
    print("  🏭 PLANET FACTORY: DIGITAL TWIN ONLINE")
    print("═" * 70)

    # 1. Initialize our SNN Brains
    norm = FeatureNormalizer(NORM_JSON)
    holosyn = HoloSynHeads()
    student_hf = StudentDistilledHeadsHF()
    student_basic = StudentDistilledHeadsBasic()
    perception = PerceptionPipeline(norm, holosyn, student_hf, student_basic)

    # 2. Spawn the AI Machine and Safety Governor
    ai_arm = IntelligentMachine("Assembly-Arm", "robot_arm", perception)
    governor = SafetyGovernor(IndustrialThresholds(max_temp=85.0))

    # 3. Connect to the 3D Physics Environment
    env = FactoryDigitalTwin()

    print("\n🚀 Commencing Physical Embodiment Loop...")

    try:
        cycle = 0
        while True: # Run until user closes the window
            cycle += 1

            # --- SENSE ---
            # Extract live physics data
            live_img = env.get_robot_camera_feed()
            live_haptics = env.get_joint_states()

            # Package into ModalityBundle
            bundle = ModalityBundle(
                E_t_aug=torch.randn(1, 528), # Mock text command
                E_i=torch.from_numpy(live_img).unsqueeze(0).flatten(1)[:, :2048], # Sliced Vision output
                E_a=torch.randn(1, 2048),
                E_v=torch.randn(1, 2048),
                H=torch.from_numpy(live_haptics).unsqueeze(0),
                feat_hf=torch.randn(1, 789),
                feat_basic=torch.randn(1, 5),
                mask=torch.ones(1, 4),
            )

            # --- THINK & SAFETY CHECK ---
            # Pass physical data through the Spiking Neural Network
            fused_emb, _, _ = ai_arm.perception.perceive(bundle)
            proposed_id = ai_arm.action_head.predict(fused_emb)

            # Validate against our physical digital twin temperature
            metrics = {"temperature": env.simulated_temp}
            is_safe, reason = governor.validate_action(proposed_id, metrics)

            if is_safe:
                final_action = proposed_id
                action_name = ai_arm.domain_cfg["action_map"].get(final_action, "Unknown")
            else:
                final_action = -1 # Triggers the cool-down rest state in apply_action
                action_name = f"🛑 BLOCKED ({reason})"

            # --- ACT ---
            # Execute in the physics engine and update the HUD
            env.apply_action(final_action)
            env.update_hud(action_name, is_safe, env.simulated_temp)

            if cycle % 100 == 0:
                print(f"  [Cycle {cycle}] Temp: {env.simulated_temp:.1f}C -> AI Decided: {action_name}")

            # Slow down loop slightly so it's viewable by human eyes
            time.sleep(1./240.)

    except KeyboardInterrupt:
        print("\n🛑 Simulation Terminated by User.")
        p.disconnect()

if __name__ == "__main__":
    run_embodied_simulation()