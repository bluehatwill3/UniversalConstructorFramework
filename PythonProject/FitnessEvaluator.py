class FitnessEvaluator:
    """Tracks and scores robot performance based on real-world industrial KPIs."""

    def __init__(self):
        self.performance_history = defaultdict(list)

    def calculate_fitness(self, robot: 'SelfReplicatingRobot') -> float:
        """
        Fitness = (Materials Gathered + Parts Produced) / (Energy Consumed + 1)
        High fitness indicates efficiency in mining or production.
        """
        if not robot.action_log:
            return 0.0

        total_produced = robot.resources.component_parts
        total_energy_spent = sum([log.get('energy_delta', 0.3) for log in robot.action_log])

        # Domain-specific weighting
        multiplier = 1.2 if robot.domain in ("asteroid_factory", "cnc_production") else 1.0

        fitness_score = (total_produced * multiplier) / (total_energy_spent + 1e-6)
        return round(fitness_score, 4)


# ── Updated Replication with Genetic Selection ─────────────────────────────

def replicate_with_fitness(self, evaluator: FitnessEvaluator, mutation_power: float = 0.05):
    fitness = evaluator.calculate_fitness(self)

    # Only allow replication if fitness is above a baseline or if the robot is "Alpha"
    if not self.is_child or fitness > 0.5:
        print(f"  🧬 Fitness check PASSED ({fitness}). Proceeding with distillation...")

        # Higher fitness reduces mutation power (preserving stable, winning traits)
        # Lower fitness increases mutation power (searching for better strategies)
        dynamic_mutation = mutation_power / (fitness + 0.1)

        return self.replicate(mutation_power=np.clip(dynamic_mutation, 0.01, 0.2))
    else:
        print(f"  ⚠️ Fitness too low ({fitness}). Replication aborted to prevent inefficient drift.")
        return None