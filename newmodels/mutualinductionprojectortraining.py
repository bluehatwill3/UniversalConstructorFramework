"""
╔══════════════════════════════════════════════════════════════════════════════╗
║             MAGNETO-OPTIC CV: MUTUAL INDUCTION TRAINING PIPELINE           ║
║                                                                              ║
║  Trains two Projector Models in parallel on real-world datasets.             ║
║  Applies Faraday's Mutual Induction to the backpropagation loop: the latent  ║
║  variance (flux) of Node A dynamically scales the learning loss of Node B.   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
import random

try:
    from sklearn.datasets import load_digits
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ─── CONFIGURATION ───
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 256
BATCH_SIZE = 64
EPOCHS = 15

# ═════════════════════════════════════════════════════════════════════════════
#  1. REAL-WORLD DATASET PIPELINE (Defect Simulation)
# ═════════════════════════════════════════════════════════════════════════════

def get_real_world_dataloaders():
    """Fetches complex real-world textures to simulate heavy glass inspection."""
    try:
        import torchvision
        import torchvision.transforms as transforms

        print("🌐 Fetching Open-Source Texture Data (FashionMNIST as proxy)...")
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Two streams simulate two robots looking at different materials.
        dataset_a = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        dataset_b = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

        loader_a = DataLoader(dataset_a, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        loader_b = DataLoader(dataset_b, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        return loader_a, loader_b
    except Exception as e:
        print(f"⚠️ Torchvision unavailable ({e}). Falling back to local digits proxy dataset.")

    if HAS_SKLEARN:
        digits = load_digits()
        x = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1) / 16.0  # [N,1,8,8]
        x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        x = (x - 0.5) / 0.5
        y = torch.zeros((x.shape[0],), dtype=torch.long)

        split = int(0.8 * x.shape[0])
        dataset_a = TensorDataset(x[:split], y[:split])
        dataset_b = TensorDataset(x[split:], y[split:])
    else:
        print("⚠️ scikit-learn unavailable. Using synthetic grayscale textures.")
        x = torch.randn(2000, 1, 128, 128)
        y = torch.zeros((2000,), dtype=torch.long)
        dataset_a = TensorDataset(x[:1600], y[:1600])
        dataset_b = TensorDataset(x[1600:], y[1600:])

    loader_a = DataLoader(dataset_a, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    loader_b = DataLoader(dataset_b, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return loader_a, loader_b

def inject_faraday_defects(img_batch):
    """Synthetically induces optical fractures into the real-world batch."""
    batch_size = img_batch.size(0)
    labels = torch.zeros(batch_size, 1).to(DEVICE)
    
    # 30% chance for any given image to have a physical fracture
    for i in range(batch_size):
        if random.random() < 0.30:
            labels[i] = 1.0
            # Induce a harsh zero-out fracture (simulating scattered polarized light)
            x_start = random.randint(20, 80)
            y_start = random.randint(20, 80)
            img_batch[i, 0, y_start:y_start+25, x_start:x_start+5] = -1.0 # Vertical crack
            img_batch[i, 0, y_start+10:y_start+15, x_start-10:x_start+15] = -1.0 # Cross crack
            
    return img_batch.to(DEVICE), labels

# ═════════════════════════════════════════════════════════════════════════════
#  2. PROJECTOR MODEL ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════

class MagnetoProjector(nn.Module):
    """Maps raw magneto-optical features to the 256-D Swarm Latent Space."""
    def __init__(self, output_dim=256):
        super(MagnetoProjector, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, output_dim)
        )
        # Temporary classification head just for training the latent space
        self.defect_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(output_dim, 1)
        )

    def forward(self, x):
        latent = self.projector(self.backbone(x))
        pred = self.defect_head(latent)
        return latent, pred

# ═════════════════════════════════════════════════════════════════════════════
#  3. MUTUAL INDUCTION TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def train_mutual_induction():
    print("═" * 80)
    print("  ⚡ PHASE 33: MAGNETO-OPTIC MUTUAL INDUCTION TRAINING")
    print("═" * 80)
    
    loader_a, loader_b = get_real_world_dataloaders()
    
    node_alpha = MagnetoProjector(output_dim=LATENT_DIM).to(DEVICE)
    node_beta = MagnetoProjector(output_dim=LATENT_DIM).to(DEVICE)
    
    opt_alpha = optim.Adam(node_alpha.parameters(), lr=1e-3)
    opt_beta = optim.Adam(node_beta.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(reduction='none') # Keep unreduced to apply induction scalar
    
    print(f"\n🚀 Commencing Swarm Training on {DEVICE} for {EPOCHS} Epochs...")
    
    for epoch in range(1, EPOCHS + 1):
        node_alpha.train()
        node_beta.train()
        
        total_loss_a = 0; total_loss_b = 0
        correct_a = 0; correct_b = 0; total_samples = 0
        
        # Zip loaders to process simultaneous sensory experiences
        for (batch_a, _), (batch_b, _) in zip(loader_a, loader_b):
            # 1. Inject real-world physics fractures
            imgs_a, labels_a = inject_faraday_defects(batch_a)
            imgs_b, labels_b = inject_faraday_defects(batch_b)
            
            opt_alpha.zero_grad()
            opt_beta.zero_grad()
            
            # 2. Forward Passes
            latent_a, pred_a = node_alpha(imgs_a)
            latent_b, pred_b = node_beta(imgs_b)
            
            # 3. Calculate Faraday Flux (Latent Space Variance)
            # This represents how "confused" or "stimulated" the node is by its current view
            flux_a = torch.var(latent_a, dim=1).mean().item()
            flux_b = torch.var(latent_b, dim=1).mean().item()
            
            # 4. Standard Task Loss
            base_loss_a = criterion(pred_a, labels_a).mean()
            base_loss_b = criterion(pred_b, labels_b).mean()
            
            # 5. FARADAY'S MUTUAL INDUCTION
            # The loss (learning signal) of one node is multiplied by the flux of the other.
            # If Alpha sees a massive defect (high flux), Beta learns its own batch 2x faster.
            induction_scalar_a = 1.0 + min(2.0, flux_b * 0.5)
            induction_scalar_b = 1.0 + min(2.0, flux_a * 0.5)
            
            loss_a = base_loss_a * induction_scalar_a
            loss_b = base_loss_b * induction_scalar_b
            
            # 6. Backprop
            loss_a.backward()
            loss_b.backward()
            
            opt_alpha.step()
            opt_beta.step()
            
            # Tracking
            total_loss_a += loss_a.item()
            total_loss_b += loss_b.item()
            
            acc_a = ((torch.sigmoid(pred_a) > 0.5) == labels_a).sum().item()
            acc_b = ((torch.sigmoid(pred_b) > 0.5) == labels_b).sum().item()
            correct_a += acc_a
            correct_b += acc_b
            total_samples += BATCH_SIZE
            
        epoch_acc_a = (correct_a / total_samples) * 100
        epoch_acc_b = (correct_b / total_samples) * 100
        
        print(f"[Epoch {epoch:02d}/{EPOCHS}]")
        print(f"  -> Node Alpha | Acc: {epoch_acc_a:5.1f}% | Inductive Loss: {total_loss_a/len(loader_a):.4f} (Driven by Beta's Flux)")
        print(f"  -> Node Beta  | Acc: {epoch_acc_b:5.1f}% | Inductive Loss: {total_loss_b/len(loader_b):.4f} (Driven by Alpha's Flux)")

    # Save the best model for the ros2_cyber_node deployment
    final_weights_path = "magneto_projector_weights.pt"
    torch.save(node_alpha.state_dict(), final_weights_path)
    print(f"\n💾 Training Complete. Collective Knowledge encoded and saved to '{final_weights_path}'.")

if __name__ == "__main__":
    train_mutual_induction()