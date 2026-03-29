"""
╔══════════════════════════════════════════════════════════════════════════════╗
║             MAGNETO-OPTIC CV: MUTUAL INDUCTION TRAINING PIPELINE           ║
║                                                                              ║
║  Trains two Projector Models in parallel on real-world lens data.            ║
║  Applies Faraday's Mutual Induction: the latent variance (Flux) of Node A    ║
║  dynamically scales the Backpropagation gradient of Node B.                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import random
from PIL import Image

try:
    import torchvision
    import torchvision.transforms as transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# ─── CONFIGURATION ───
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 256
BATCH_SIZE = 64
EPOCHS = 15

# ═════════════════════════════════════════════════════════════════════════════
#  1. FARADAY LENS DATA LAKE (Real World + High-Fidelity Proxy)
# ═════════════════════════════════════════════════════════════════════════════

class FaradayLensLake:
    """Ingests real-world lens images or simulates borosilicate glass physics."""
    def __init__(self, data_dir="lens_data"):
        self.data_dir = os.path.join("/home/scidev/PycharmProjects/PythonProject/", data_dir)
        self.real_images = []
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]) if HAS_TORCHVISION else None
        
        if os.path.exists(self.data_dir) and HAS_TORCHVISION:
            valid_exts = ('.png', '.jpg', '.jpeg')
            for f in os.listdir(self.data_dir):
                if f.lower().endswith(valid_exts):
                    img = Image.open(os.path.join(self.data_dir, f)).convert('L')
                    self.real_images.append(self.transform(img))
        
        if self.real_images:
            print(f"✅ Faraday Lake: Loaded {len(self.real_images)} real lens textures.")
        else:
            print("⚠️ Faraday Lake: No local images. Initializing borosilicate simulation.")

    def generate_inductive_batch(self, batch_size):
        imgs = []
        labels = []
        for _ in range(batch_size):
            is_defective = random.random() < 0.25
            if self.real_images:
                img = random.choice(self.real_images).clone()
            else:
                img = torch.randn(1, 128, 128) * 0.1 # Base grain
            
            if is_defective:
                # Simulate "Shatter" aberration (Faraday Optics)
                x, y = random.randint(30, 90), random.randint(30, 90)
                img[:, y-10:y+10, x-1:x+1] = -1.0 # Micro-fracture
                labels.append(1.0)
            else:
                labels.append(0.0)
            imgs.append(img)
            
        return torch.stack(imgs).to(DEVICE), torch.tensor(labels).unsqueeze(1).to(DEVICE)

# ═════════════════════════════════════════════════════════════════════════════
#  2. PROJECTOR MODEL (Resonant Architecture)
# ═════════════════════════════════════════════════════════════════════════════

class MagnetoProjector(nn.Module):
    def __init__(self, output_dim=256):
        super(MagnetoProjector, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        self.defect_head = nn.Linear(output_dim, 1)

    def forward(self, x):
        latent = self.projector(self.backbone(x))
        pred = self.defect_head(latent)
        return latent, pred

# ═════════════════════════════════════════════════════════════════════════════
#  3. MUTUAL INDUCTION TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def run_inductive_training():
    print("═" * 80)
    print("  ⚡ PHASE 33: FARADAY MAGNETO-OPTIC MUTUAL INDUCTION")
    print("═" * 80)
    
    lake = FaradayLensLake()
    node_alpha = MagnetoProjector().to(DEVICE)
    node_beta = MagnetoProjector().to(DEVICE)
    
    opt_a = optim.Adam(node_alpha.parameters(), lr=1e-3)
    opt_b = optim.Adam(node_beta.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(1, EPOCHS + 1):
        total_a_loss = 0; total_b_loss = 0
        correct_a = 0; correct_b = 0
        
        # 100 iterations per epoch to simulate "all data"
        for _ in range(100):
            imgs_a, labels_a = lake.generate_inductive_batch(BATCH_SIZE)
            imgs_b, labels_b = lake.generate_inductive_batch(BATCH_SIZE)
            
            opt_a.zero_grad(); opt_b.zero_grad()
            
            # Forward Pass
            lat_a, pred_a = node_alpha(imgs_a)
            lat_b, pred_b = node_beta(imgs_b)
            
            # MUTUAL INDUCTION: Calculate Faraday Flux (Latent Variance)
            flux_a = torch.std(lat_a).item()
            flux_b = torch.std(lat_b).item()
            
            # Cross-Induce gradients: alpha learns harder when beta's flux is high
            loss_a = criterion(pred_a, labels_a) * (1.0 + flux_b * 2.0)
            loss_b = criterion(pred_b, labels_b) * (1.0 + flux_a * 2.0)
            
            loss_a.backward(); loss_b.backward()
            opt_a.step(); opt_b.step()
            
            total_a_loss += loss_a.item(); total_b_loss += loss_b.item()
            correct_a += ((torch.sigmoid(pred_a) > 0.5) == labels_a).sum().item()
            correct_b += ((torch.sigmoid(pred_b) > 0.5) == labels_b).sum().item()

        acc_a = (correct_a / (100 * BATCH_SIZE)) * 100
        acc_b = (correct_b / (100 * BATCH_SIZE)) * 100
        
        print(f"[Epoch {epoch:02d}] Alpha Acc: {acc_a:5.1f}% | Beta Acc: {acc_b:5.1f}% | Flux induced by B: {flux_b:.4f}")

    torch.save(node_alpha.state_dict(), "magneto_projector_weights.pt")
    print("\n💾 Training Complete. Inducted weights saved for Phase 34 Deployment.")

if __name__ == "__main__":
    run_inductive_training()