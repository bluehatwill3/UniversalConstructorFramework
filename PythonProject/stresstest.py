import torch
import numpy as np
from datetime import datetime
from typing import Any

# 1. Correct the Alias Import
from ucf1 import UniversalConstructor, SelfReplicatingRobot as IndustrialRobot
from SafetyGovernor import IndustrialThresholds, SafetyGovernor
from IndustrialAnalytics import IndustrialAnalytics

def run_stress_test(constructor: 'UniversalConstructor', crisis_type: str = "thermal"):
    print(f"\n🔥 INITIATING STRESS TEST: {crisis_type.upper()} CRISIS")
    print("═" * 70)

    # Note: Using .values() if constructor.robots is a dictionary
    robots_list = list(constructor.robots.values()) if isinstance(constructor.robots, dict) else constructor.robots

    for cycle in range(1, 11):
        for robot in robots_list:
            # 1. Crisis Injection
            if crisis_type == "thermal":
                temp = 95.0 + (np.random.rand() * 5)
                vib = 0.3
            else:
                temp = 70.0
                vib = 1.5 + (np.random.rand() * 0.5)

            # 2. Perception & Action Proposal
            state = torch.randn(1, 256)
            proposed_id = robot.action_head.predict(state)

            # 3. Safety Verification
            metrics = {"temp": temp, "vib": vib, "energy": robot.energy}
            # Note: Ensure your Governor class uses these keys
            is_safe, reason = constructor.governor.validate_action(proposed_id, metrics)

            # 4. Enforce Logic
            final_id = proposed_id if is_safe else 0
            status = "✅ SAFE" if is_safe else f"🛑 BLOCKED ({reason})"

            # Update robot state and log
            robot.energy -= 0.5
            robot.action_log.append({
                "cycle": cycle,
                "action_id": final_id,
                "action_name": status,
                "safe": is_safe,
                "reason": reason
            })

            print(f" Cycle {cycle:02d} | {robot.name:<15} | Sensor: {temp:.1f}°C / {vib:.2f}G | {status}")

            # 5. Corrective Retraining Trigger
            if hasattr(constructor, 'check_for_retraining'):
                constructor.check_for_retraining(robot)

    # Export for post-mortem analysis
    IndustrialAnalytics.export_hive_logs(robots_list, filename="stress_test_report.csv")


if __name__ == "__main__":
    # Initialize factory with safety systems
    factory = UniversalConstructor(n_cycles=10)
    run_stress_test(factory, crisis_type="thermal")