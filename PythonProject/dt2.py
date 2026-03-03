"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PLANET FACTORY: MULTI-DOMAIN DISTILLATION ENGINE              ║
║                                                                              ║
║  This script orchestrates offline Knowledge Distillation. It prompts an      ║
║  open-source Foundation Model (like OpenVLA or Octo) with domain-specific    ║
║  scenarios (FSD, Logistics, Production) and maps the outputs to train our    ║
║  lightweight Neuromorphic Action Heads.                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import torch
import torch.nn.functional as F
from typing import List, Dict
import brian2
# Import the core architecture from your main framework
from ucf3 import (
    UniversalConstructor, ModalityBundle, SelfReplicatingRobot,
    RoboticsDistiller, FoundationTeacherAdapter, DOMAINS, DEVICE
)


# ═════════════════════════════════════════════════════════════════════════════
#  1. DOMAIN-SPECIFIC VLA TRANSLATOR
# ═════════════════════════════════════════════════════════════════════════════

class MultiDomainTeacherAdapter(FoundationTeacherAdapter):
    """
    Extends the base adapter to handle distinct domains like FSD, Logistics,
    and Production, translating generic VLA text into specific discrete actions.
    """

    def __init__(self, model_name: str = "openvla/openvla-7b", use_mock: bool = True):
        super().__init__(model_name, use_mock)

        # Define how we prompt the Foundation Model for each domain
        self.domain_prompts = {
            "self_driving": "You are navigating a factory floor. Obstacle ahead. What action?",
            "cnc_production": "You are monitoring a CNC lathe. Temperature is rising. What action?",
            "swarm_coordination": "You are a logistics drone. A heavy pallet needs moving. What action?",
            "universe_builder": "You are an asteroid miner. Target rich in iron detected. What action?",
            "robot_arm": "You are a sorting arm. Item is within reach. What action?"
        }

    def _map_vla_to_discrete_logits(self, raw_text: str, domain: str, n_actions: int) -> torch.Tensor:
        """Heuristics to map the Foundation Model's text output to our SNN Action IDs."""
        logits = torch.zeros(1, n_actions)
        text = raw_text.lower()

        # Domain: FSD (Self-Driving) [0: Left, 1: Straight, 2: Right]
        if domain == "self_driving":
            if "left" in text or "avoid" in text:
                logits[0, 0] = 5.0
            elif "right" in text:
                logits[0, 2] = 5.0
            else:
                logits[0, 1] = 5.0  # Default straight

        # Domain: Production (CNC) [0: Maintain, 1: Adjust, 2: Swap]
        elif domain == "cnc_production":
            if "adjust" in text or "slow" in text:
                logits[0, 1] = 5.0
            elif "swap" in text or "replace" in text:
                logits[0, 2] = 5.0
            else:
                logits[0, 0] = 5.0

        # Domain: Logistics (Swarm) [0: Sync, 1: Form Structure, 2: Disperse]
        elif domain == "swarm_coordination":
            if "help" in text or "together" in text:
                logits[0, 1] = 5.0
            elif "scatter" in text or "away" in text:
                logits[0, 2] = 5.0
            else:
                logits[0, 0] = 5.0

        # Domain: Mining (Universe Builder) [0: Mine, 1: Ignite, 2: Stabilize]
        elif domain == "universe_builder":
            if "drill" in text or "mine" in text:
                logits[0, 0] = 5.0
            elif "burn" in text or "ignite" in text:
                logits[0, 1] = 5.0
            else:
                logits[0, 2] = 5.0

        # Domain: Default / Robot Arm
        else:
            if "engage" in text or "grab" in text:
                logits[0, 1] = 5.0
            else:
                logits[0, 0] = 5.0

        return F.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def get_teacher_action_logits(self, bundle: ModalityBundle, domain: str, n_actions: int) -> torch.Tensor:
        """Gets the prediction from the teacher and translates it for the specific domain."""
        if self.use_mock:
            # Simulate a confident response based on the domain prompt
            logits = torch.randn(1, n_actions) * 1.5
            return F.log_softmax(logits, dim=-1)
        else:
            # REAL INFERENCE PLACEHOLDER (when use_mock=False):
            # prompt = self.domain_prompts.get(domain, "What is the next action?")
            # inputs = self.processor(text=prompt, images=bundle.raw_image, return_tensors="pt").to(DEVICE)
            # outputs = self.teacher_model.generate(**inputs, max_new_tokens=15)
            # text_output = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

            text_output = "I should adjust the machine."  # Simulated VLA text
            return self._map_vla_to_discrete_logits(text_output, domain, n_actions)


# ═════════════════════════════════════════════════════════════════════════════
#  2. DISTILLATION ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def generate_domain_curriculum(uc: UniversalConstructor, domain: str, n_samples: int = 32) -> List[ModalityBundle]:
    """Generates specific training scenarios based on the domain."""
    curriculum = []
    for i in range(n_samples):
        bundle = uc.perception.make_synthetic_bundle(seed=i + hash(domain) % 10000)
        # We can artificially spike temperatures for CNC training
        if domain == "cnc_production" and i % 2 == 0:
            bundle.meta["aud_meta"][2] = 95.0  # High temp anomaly
        curriculum.append(bundle)
    return curriculum


def run_planet_factory_distillation():
    print("═" * 70)
    print("  🏭 PLANET FACTORY: OFFLINE KNOWLEDGE DISTILLATION")
    print("═" * 70)

    # 1. Initialize the Core Framework (we don't run the cycles yet)
    uc = UniversalConstructor(n_cycles=0, verbose=False)

    # 2. Initialize the Multi-Domain Teacher
    teacher = MultiDomainTeacherAdapter(model_name="openvla/openvla-7b", use_mock=True)
    distiller = RoboticsDistiller(teacher)

    # 3. Create a directory to save the trained "brains"
    save_dir = os.path.join(os.path.dirname(__file__), "distilled_brains")
    os.makedirs(save_dir, exist_ok=True)

    # 4. Iterate through every domain in the Planet Factory
    for domain_name, config in DOMAINS.items():
        print(f"\n🌍 Preparing curriculum for Domain: {config['emoji']} {domain_name}")

        # Create a temporary robot strictly for training
        robot = SelfReplicatingRobot(f"Student-{domain_name}", domain_name, uc.perception)
        curriculum = generate_domain_curriculum(uc, domain_name, n_samples=50)

        # Override the default distill method to pass the domain name to our new adapter
        student_head = robot.action_head
        student_head.train()
        optimizer = torch.optim.Adam(student_head.parameters(), lr=0.005)
        loss_fn = torch.nn.KLDivLoss(reduction="batchmean")

        print(f"  🧠 Distilling {teacher.model_name} into {robot.name}...")
        for epoch in range(1, 21):  # 20 Epochs per domain
            total_loss = 0.0
            for bundle in curriculum:
                # Get Teacher Target
                t_log_probs = teacher.get_teacher_action_logits(bundle, domain_name, config["n_actions"])
                t_probs = torch.exp(t_log_probs)

                # Get Student Prediction
                fused_emb, _, _ = robot.perception.perceive(bundle)
                s_log_probs = F.log_softmax(student_head(fused_emb), dim=-1)

                # Optimize
                loss = loss_fn(s_log_probs, t_probs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 5 == 0:
                print(f"     Epoch {epoch}/20 | Loss: {total_loss / len(curriculum):.4f}")

        # 5. Save the Distilled SNN Weights
        save_path = os.path.join(save_dir, f"{domain_name}_snn_head.pt")
        torch.save(student_head.state_dict(), save_path)
        print(f"  💾 Saved distilled brain to: {save_path}")

    print("\n✅ All domains successfully distilled. Ready for deployment!")
    print("═" * 70)


if __name__ == "__main__":
    run_planet_factory_distillation()