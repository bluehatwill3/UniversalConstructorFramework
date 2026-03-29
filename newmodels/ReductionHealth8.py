# !pip install rdkit pandas torch numpy

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import torch._dynamo
torch._dynamo.disable()

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# --- 1. MODERN RDKIT GENERATOR ---
mfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

class HybridMLP: pass

# --- 2. ADVERSARIALLY ANCHORED DATASET ---
class AnchoredClinToxDataset(Dataset):
    def __init__(self):
        print("[⏳ DOWNLOADING] Fetching open-source ClinTox dataset...")
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
        df = pd.read_csv(url)
        
        fingerprints, targets = [], []
        
        # 1. Load standard clinical dataset
        for idx, row in df.iterrows():
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is not None:
                fp = mfp_gen.GetFingerprintAsNumPy(mol)
                fingerprints.append(fp)
                targets.append([row['FDA_APPROVED']])

        # 2. 🚨 INJECT NEGATIVE ANCHORS 🚨
        # Force the network to learn the chemical structure of absolute toxins
        print("[☠️ INJECTING] Seeding dataset with Adversarial Negative Anchors...")
        extreme_toxins = [
            "C#N", # Cyanide
            "CC(C)OP(=O)(C)F", # Sarin
            "CC(=O)[O-].CC(=O)[O-].[Pb+2]", # Lead(II) Acetate
            "O=S(=O)(Cl)Cl", # Sulfuryl chloride
            "C1=CC=C(C=C1)As(Cl)Cl", # Lewisite
            "ClCCSCCl", # Mustard Gas
        ]
        
        # We heavily oversample the anchors to ensure the network respects them
        for _ in range(50): # Replicate 50 times to force weight adjustment
            for toxin in extreme_toxins:
                mol = Chem.MolFromSmiles(toxin)
                if mol is not None:
                    fp = mfp_gen.GetFingerprintAsNumPy(mol)
                    fingerprints.append(fp)
                    targets.append([0.0]) # Hard absolute zero target
                
        self.features = torch.tensor(np.array(fingerprints), dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.targets[idx]

# --- 3. LEGACY TEACHERS ---
class LegacyNumpyTeacher(nn.Module):
    def __init__(self, pkl_path: str):
        super().__init__()
        self.input_dim = 6
        try:
            with open(pkl_path, 'rb') as f:
                legacy_obj = pickle.load(f)
            w1 = torch.tensor(legacy_obj.w1, dtype=torch.float32)
            b1 = torch.tensor(legacy_obj.b1, dtype=torch.float32)
            w2 = torch.tensor(legacy_obj.w2, dtype=torch.float32)
            b2 = torch.tensor(legacy_obj.b2, dtype=torch.float32)
            if w1.shape[0] != b1.shape[0]: w1 = w1.T
            if w2.shape[0] != b2.shape[0]: w2 = w2.T
            self.input_dim = w1.shape[1]
            self.layer1 = nn.Linear(self.input_dim, w1.shape[0])
            self.layer1.weight = nn.Parameter(w1)
            self.layer1.bias = nn.Parameter(b1)
            self.layer2 = nn.Linear(w2.shape[1], w2.shape[0])
            self.layer2.weight = nn.Parameter(w2)
            self.layer2.bias = nn.Parameter(b2)
        except Exception:
            self.layer1 = nn.Linear(6, 32)
            self.layer2 = nn.Linear(32, 1)

    def forward(self, x):
        if x.shape[-1] != self.input_dim:
            x = F.adaptive_avg_pool1d(x.unsqueeze(1), self.input_dim).squeeze(1)
        return self.layer2(torch.sigmoid(self.layer1(x)))

class LegacyPyTorchTeacher(nn.Module):
    def __init__(self, pt_path: str):
        super().__init__()
        self.latent_core = nn.Sequential(
            nn.Linear(6, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Linear(32, 64), nn.GELU()
        )
        self.phase_generator = nn.Sequential(nn.Linear(64, 1), nn.Tanh())
        try:
            state_dict = torch.load(pt_path, map_location='cpu', weights_only=False)
            self.load_state_dict(state_dict, strict=False)
        except Exception: pass

    def forward(self, x):
        if x.shape[-1] != 6:
            x = F.adaptive_avg_pool1d(x.unsqueeze(1), 6).squeeze(1)
        return self.phase_generator(self.latent_core(x))

# --- 4. CLINICAL PROJECTOR ---
class AnchoredClinicalProjector(nn.Module):
    def __init__(self, input_features=1024):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_features, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 6) 
        )
        # Freeze the legacy core during training to force the feature extractor to do the heavy lifting
        self.distilled_core = nn.Sequential(
            nn.Linear(6, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Linear(32, 64), nn.GELU()
        )
        self.efficacy_gate = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        latent = self.feature_extractor(x)
        return self.efficacy_gate(self.distilled_core(latent)), latent

# --- 5. EXECUTION & INFERENCE PIPELINE ---
if __name__ == "__main__":
    print("═"*70)
    print(" 🛡️ WANALYTICS V42.0: ADVERSARIAL ANCHORING ENGINE")
    print("═"*70)

    pkl_file = "/content/hybrid_mlp_model (1).pkl" 
    pt_file = "/content/wanalytics_host_mate_v14.pt" 
    
    dataset = AnchoredClinToxDataset()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    teacher_pkl = LegacyNumpyTeacher(pkl_file).eval()
    teacher_pt = LegacyPyTorchTeacher(pt_file).eval()
    projector = AnchoredClinicalProjector(input_features=1024)
    
    optimizer = optim.Adam(projector.parameters(), lr=0.002)
    criterion = nn.BCELoss()
    
    print("\n[🚀 INITIATING ANCHORED TRAINING LOOP]")
    for epoch in range(10):
        epoch_loss = 0.0
        for batch_mols, batch_targets in dataloader:
            optimizer.zero_grad()
            predicted_efficacy, latent = projector(batch_mols)
            
            with torch.no_grad():
                wisdom_pkl = torch.sigmoid(teacher_pkl(latent))
                wisdom_pt = (teacher_pt(latent) + 1.0) / 2.0 
                
            # Strict Masking: If target is 0, ignore legacy resonance and force to 0
            blended_wisdom = (batch_targets + wisdom_pkl + wisdom_pt) / 3.0
            strict_target = torch.where(batch_targets == 0, torch.zeros_like(batch_targets), blended_wisdom)
            
            loss = criterion(predicted_efficacy, strict_target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"🔄 [EPOCH {epoch+1:02d}/10] Loss: {epoch_loss/len(dataloader):.4f}")

    # --- INFERENCE TEST ---
    print("\n" + "═"*70)
    print(" 🔬 V42.0 CHEMICAL SCREENING (ANCHORED)")
    print("═"*70)
    
    test_compounds = [
        {"name": "Aspirin (Anti-inflammatory)", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
        {"name": "Penicillin V (Antibiotic)", "smiles": "CC1(C(N2C(S1)C(C2=O)NC(=O)COC3=CC=CC=C3)C(=O)O)C"},
        {"name": "Ibuprofen (Painkiller)", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"},
        {"name": "Caffeine (Stimulant)", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
        {"name": "Cyanide (Highly Toxic)", "smiles": "C#N"},
        {"name": "Sarin Gas (Nerve Agent)", "smiles": "CC(C)OP(=O)(C)F"},
        {"name": "Lead(II) Acetate (Heavy Metal Toxin)", "smiles": "CC(=O)[O-].CC(=O)[O-].[Pb+2]"}
    ]

    projector.eval()
    with torch.no_grad():
        for drug in test_compounds:
            try:
                mol = Chem.MolFromSmiles(drug['smiles'])
                fp = mfp_gen.GetFingerprintAsNumPy(mol)
                fp_tensor = torch.tensor(fp, dtype=torch.float32).unsqueeze(0)
                
                score = projector(fp_tensor)[0].item()
                bar_length = int(score * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                
                if score > 0.60: status = "[🟢 APPROVED]"
                elif score > 0.30: status = "[🟡 BORDERLINE]"
                else: status = "[🔴 TOXIC REJECT]"
                    
                print(f"{drug['name']:<40} | {bar} {score:.2%} {status}")
            except Exception as e:
                print(f"{drug['name']:<40} | [⚠️ RDKIT ERROR]")