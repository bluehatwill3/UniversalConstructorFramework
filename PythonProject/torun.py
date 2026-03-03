# All 6 domains, 60 cycles
python universal_constructor.py --model-dir ./models --cycles 60

# Specific domains, quiet mode, export log
python universal_constructor.py --domains robot_arm universe_builder --cycles 100 --quiet --export hive_log.json

# Standalone embedding
from universal_constructor import load_production_stack
pipeline, info = load_production_stack("./models")
bundle = pipeline.make_synthetic_bundle(available_modalities=["text","audio"])
emb, weights, details = pipeline.perceive(bundle)