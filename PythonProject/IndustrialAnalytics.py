import csv
import os
from datetime import datetime
from typing import List, Dict, Any  # Added the missing List import


# Placeholder or Import for the Robot class to avoid NameError
class IndustrialRobot:
    def __init__(self, name: str, domain: str):
        self.name = name
        self.domain = domain
        self.action_log = []


class IndustrialAnalytics:
    """Handles the extraction and formatting of hive data for external reporting."""

    @staticmethod
    def export_hive_logs(robots: List[IndustrialRobot], filename: str = "factory_report.csv"):
        """Exports the complete action and safety history of the robot hive."""

        headers = [
            "timestamp", "robot_name", "domain", "cycle",
            "action_executed", "is_safe", "override_reason",
            "energy_level", "parts_produced"
        ]

        try:
            with open(filename, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=headers)
                writer.writeheader()

                for robot in robots:
                    cumulative_parts = 0
                    for i, log in enumerate(robot.action_log):
                        if log.get("safe", False):
                            cumulative_parts += 1

                        writer.writerow({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "robot_name": robot.name,
                            "domain": robot.domain,
                            "cycle": i + 1,
                            "action_executed": log.get("action_name", "UNKNOWN"),
                            "is_safe": log.get("safe", True),
                            "override_reason": log.get("reason", "NONE"),
                            "energy_level": round(100 - (i * 0.5), 2),
                            "parts_produced": cumulative_parts
                        })
            print(f"\n📊 Analytics Export Complete: {os.path.abspath(filename)}")
        except Exception as e:
            print(f"❌ Export Failed: {e}")