# !pip install rdkit torch pandas numpy

import os
# Force PyTorch to disable Dynamo/SymPy background compilation hooks
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import torch._dynamo
torch._dynamo.disable()

import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

# --- 1. PROJECTOR CLASS DEFINITION (Required to load the .pt file) ---
class ClinicalDrugProjector(nn.Module):
    """
    Evaluates real 1024-D molecular fingerprints for biological viability.
    """
    def __init__(self, input_features=1024):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_features, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 6) 
        )
        
        self.distilled_core = nn.Sequential(
            nn.Linear(6, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Linear(32, 64), nn.GELU()
        )
        
        self.efficacy_gate = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

    def forward(self, molecular_fingerprint):
        latent_features = self.feature_extractor(molecular_fingerprint)
        core_processing = self.distilled_core(latent_features)
        efficacy = self.efficacy_gate(core_processing)
        return efficacy

# --- 2. SMILES PROCESSING PIPELINE ---
def generate_fingerprint(smiles_string: str) -> torch.Tensor:
    """Converts a chemical string into a 1024-bit Morgan Fingerprint."""
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles_string}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return torch.tensor(list(fp), dtype=torch.float32).unsqueeze(0) # Add batch dimension

# --- 3. REAL-WORLD DRUG TEST BATCH ---
test_compounds = [
    {"name": "Aspirin (Anti-inflammatory)", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "type": "Safe"},
    {"name": "Penicillin V (Antibiotic)", "smiles": "CC1(C(N2C(S1)C(C2=O)NC(=O)COC3=CC=CC=C3)C(=O)O)C", "type": "Safe"},
    {"name": "Ibuprofen (Painkiller)", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "type": "Safe"},
    {"name": "Caffeine (Stimulant)", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "type": "Safe"},
    {"name": "Cyanide (Highly Toxic)", "smiles": "C#N", "type": "Toxic"},
    {"name": "Sarin Gas (Nerve Agent)", "smiles": "CC(C)OP(=O)(C)F", "type": "Toxic"},
    {"name": "Lead(II) Acetate (Heavy Metal Toxin)", "smiles": "CC(=O)[O-].CC(=O)[O-].[Pb+2]", "type": "Toxic"}
]

# --- 4. EXECUTION BOOT SEQUENCE ---
if __name__ == "__main__":
    print("═"*70)
    print(" 🔬 WANALYTICS V40.0: CLINICAL INFERENCE ENGINE")
    print(" ⚙️  STATUS: Loading 'wanalytics_clinical_projector.pt'")
    print("═"*70)

    # Load the trained model
    model_path = "wanalytics_clinical_projector.pt"
    
    try:
        projector = ClinicalDrugProjector(input_features=1024)
        projector.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        projector.eval()
        print("[✅ MODEL MOUNTED] Ready for chemical screening.\n")
    except Exception as e:
        print(f"[💀 FATAL ERROR] Could not load model: {e}")
        exit()

    print(f"{'COMPOUND NAME':<40} | {'PREDICTED EFFICACY / VIABILITY'}")
    print("-" * 70)
    
    results = []
    
    with torch.no_grad():
        for drug in test_compounds:
            try:
                # 1. Convert to binary fingerprint
                fingerprint = generate_fingerprint(drug['smiles'])
                
                # 2. Run through Wanalytics Projector
                efficacy_tensor = projector(fingerprint)
                score = efficacy_tensor.item()
                
                # 3. Format output
                bar_length = int(score * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                
                if score > 0.70:
                    status = "[🟢 APPROVED]"
                elif score > 0.40:
                    status = "[🟡 BORDERLINE]"
                else:
                    status = "[🔴 REJECTED / TOXIC]"
                    
                print(f"{drug['name']:<40} | {bar} {score:.2%} {status}")
                results.append({"name": drug['name'], "score": score})
                
            except Exception as e:
                print(f"{drug['name']:<40} | [⚠️ RDKIT PARSE ERROR]")

    print("\n--- ANALYSIS COMPLETE ---")
    best_drug = max(results, key=lambda x: x['score'])
    print(f"🏆 Top Candidate for Ecosystem Deployment: {best_drug['name']} ({best_drug['score']:.2%})")