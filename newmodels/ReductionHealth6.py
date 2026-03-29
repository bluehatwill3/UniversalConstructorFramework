# !pip install rdkit pandas torch numpy

import os
# Force PyTorch to disable Dynamo/SymPy background compilation hooks
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
from rdkit.Chem import AllChem

# --- 1. PICKLE NAMESPACE MAPPING ---
class HybridMLP:
    """Blueprint for pickle to reconstruct the legacy Numpy object."""
    pass

# --- 2. REAL-WORLD OPEN-SOURCE DRUG DATASET (CLINTOX) ---
class RealWorldClinToxDataset(Dataset):
    """
    Downloads the open-source ClinTox dataset from MoleculeNet.
    Converts SMILES strings into 1024-bit Morgan Fingerprints using RDKit.
    Targets 'FDA_APPROVED' status as the biological viability metric.
    """
    def __init__(self):
        print("[⏳ DOWNLOADING] Fetching open-source ClinTox dataset from MoleculeNet AWS...")
        # Direct link to the open-source ClinTox dataset
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
        df = pd.read_csv(url)
        
        fingerprints = []
        targets = []
        
        print("[🧪 PROCESSING] Converting SMILES strings to 1024-bit Morgan Fingerprints...")
        for idx, row in df.iterrows():
            smiles = row['smiles']
            # Target is 1 if FDA Approved, 0 if Failed Clinical Trials
            target = row['FDA_APPROVED'] 
            
            # Use RDKit to parse the chemical structure
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Generate a 1024-dimensional binary vector representing the molecule
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fingerprints.append(list(fp))
                targets.append([target])
                
        self.features = torch.tensor(fingerprints, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
        print(f"[✅ DATASET READY] Successfully compiled {len(self.features)} real-world chemical compounds.")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# --- 3. LEGACY MODEL MOUNTING (TEACHERS) ---
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
            print(f"[✅ PKL MOUNTED] Legacy Numpy weights loaded.")
        except Exception as e:
            print(f"[⚠️ PKL FALLBACK] Reverting to baseline: {e}")
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
            print(f"[✅ PT MOUNTED] Legacy PyTorch V14 logic loaded.")
        except Exception as e:
            print(f"[⚠️ PT FALLBACK] Reverting to baseline: {e}")

    def forward(self, x):
        if x.shape[-1] != 6:
            x = F.adaptive_avg_pool1d(x.unsqueeze(1), 6).squeeze(1)
        return self.phase_generator(self.latent_core(x))

# --- 4. REAL-WORLD DRUG DISCOVERY PROJECTOR ---
class ClinicalDrugProjector(nn.Module):
    """
    Evaluates real 1024-D molecular fingerprints for biological viability.
    """
    def __init__(self, input_features=1024):
        super().__init__()
        # Molecular dimensionality reduction (1024 -> 6D)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_features, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 6) 
        )
        
        # Distilled Latent Core (matches Wanalytics structure)
        self.distilled_core = nn.Sequential(
            nn.Linear(6, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Linear(32, 64), nn.GELU()
        )
        
        # Clinical Viability Output
        self.efficacy_gate = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid() # 1.0 = Highly Viable/Approved, 0.0 = Toxic/Failed
        )

    def forward(self, molecular_fingerprint):
        latent_features = self.feature_extractor(molecular_fingerprint)
        core_processing = self.distilled_core(latent_features)
        efficacy = self.efficacy_gate(core_processing)
        return efficacy, latent_features

# --- 5. HYBRID DISTILLATION PIPELINE ---
def train_clinical_projector(pkl_path, pt_path):
    print("═"*70)
    print(" 🔬 WANALYTICS V39.0: REAL-WORLD DRUG DISCOVERY PROJECTOR")
    print(" ⚙️  STATUS: Distilling Legacy Models onto ClinTox Open Dataset")
    print("═"*70)

    dataset = RealWorldClinToxDataset()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    teacher_pkl = LegacyNumpyTeacher(pkl_path).eval()
    teacher_pt = LegacyPyTorchTeacher(pt_path).eval()
    projector = ClinicalDrugProjector(input_features=1024)
    
    optimizer = optim.Adam(projector.parameters(), lr=0.002, weight_decay=1e-5)
    criterion = nn.BCELoss() # Binary Cross Entropy is ideal for FDA approval probability
    
    epochs = 10
    print("\n[🚀 INITIATING HYBRID TRAINING LOOP]")
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_mols, batch_targets in dataloader:
            optimizer.zero_grad()
            
            # 1. Projector evaluates the real-world molecules
            predicted_efficacy, latent_features = projector(batch_mols)
            
            # 2. Extract baseline Wanalytics math
            with torch.no_grad():
                wisdom_pkl = torch.sigmoid(teacher_pkl(latent_features))
                wisdom_pt = (teacher_pt(latent_features) + 1.0) / 2.0 # Scale tanh(-1,1) to sigmoid(0,1)
                
            # 3. Blended Loss: Model must learn real-world chemistry (batch_targets) 
            # while being constrained by the legacy network's logic
            distillation_target = (batch_targets + wisdom_pkl + wisdom_pt) / 3.0
            
            loss = criterion(predicted_efficacy, distillation_target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"🔄 [EPOCH {epoch+1:02d}/{epochs}] Clinical Distillation Loss: {epoch_loss/len(dataloader):.4f}")
        
    export_path = "wanalytics_clinical_projector.pt"
    torch.save(projector.state_dict(), export_path)
    print(f"\n[💾 MATRIX LOCKED] Real-world Clinical Projector exported to '{export_path}'.")
    return projector

# --- 6. EXECUTION ---
if __name__ == "__main__":
    # Point these to the correct locations in your environment
    pkl_file = "/workspaces/UniversalConstructorFramework/PythonProject/hybrid_mlp_model (1).pkl" 
    pt_file = "/workspaces/UniversalConstructorFramework/PythonProject/wanalytics_host_mate_v14.pt" 
    
    projector = train_clinical_projector(pkl_file, pt_file)