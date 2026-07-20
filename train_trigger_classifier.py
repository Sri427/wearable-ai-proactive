import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)

# 1. Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

CACHE_PATH = "cached_proactive_features.pt"
MODEL_SAVE_PATH = "trigger_classifier.pt"

print("=== 🎯 Training F1-Optimized Trigger Classifier ===")

# 2. Load cached features
print(f"Loading cached features from {CACHE_PATH}...")
try:
    data = torch.load(CACHE_PATH)
    print(f"Loaded {len(data)} total decision interval samples.")
except Exception as e:
    print(f"Error loading cached features: {e}")
    exit(1)

# Extract features and labels
features = torch.stack([d["feature"] for d in data]) # Shape: (N, 384)
labels = torch.tensor([d["label"] for d in data], dtype=torch.long) # Shape: (N,)

num_samples = len(labels)
num_silent = (labels == 0).sum().item()
num_interrupt = (labels == 1).sum().item()

print(f"Dataset stats: Silent=$silent$ ({num_silent}), Interrupt=$interrupt$ ({num_interrupt})")

# 3. Train/Val Split (80/20)
indices = torch.randperm(num_samples)
train_size = int(0.8 * num_samples)

train_idx, val_idx = indices[:train_size], indices[train_size:]

X_train, y_train = features[train_idx], labels[train_idx]
X_val, y_val = features[val_idx], labels[val_idx]

class FeatureDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_dataset = FeatureDataset(X_train, y_train)
val_dataset = FeatureDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 4. Model Architecture
class TriggerClassifier(nn.Module):
    def __init__(self, in_features=384, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1) # Output raw logit for binary classification
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

model = TriggerClassifier(in_features=features.shape[1]).to(device)

# Pos weight to handle precision/recall balance
pos_weight = torch.tensor([num_silent / max(1, num_interrupt)]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# 5. Training Loop
print("\n🔥 Training Neural Trigger Classifier on Apple Silicon Metal...")
best_val_f1 = 0.0
best_threshold = 0.5

epochs = 50
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device).float()
        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()
    
    # Validation
    model.eval()
    val_probs = []
    val_targets = []
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            probs = torch.sigmoid(model(bx)).cpu().numpy()
            val_probs.extend(probs)
            val_targets.extend(by.numpy())
    
    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)
    
    # Sweep threshold tau to find maximum F1
    best_ep_f1 = 0.0
    best_ep_tau = 0.5
    for tau in np.linspace(0.2, 0.8, 13):
        preds = (val_probs >= tau).astype(int)
        tp = np.sum((preds == 1) & (val_targets == 1))
        fp = np.sum((preds == 1) & (val_targets == 0))
        fn = np.sum((preds == 0) & (val_targets == 1))
        
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        
        if f1 > best_ep_f1:
            best_ep_f1 = f1
            best_ep_tau = tau
            best_ep_prec = prec
            best_ep_rec = rec

    if epoch % 10 == 0 or epoch == epochs:
        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {total_loss/len(train_loader):.4f} | Val F1: {best_ep_f1:.4f} (Prec: {best_ep_prec:.4f}, Rec: {best_ep_rec:.4f} @ tau={best_ep_tau:.2f})")

    if best_ep_f1 > best_val_f1:
        best_val_f1 = best_ep_f1
        best_threshold = best_ep_tau
        torch.save({
            "model_state": model.state_dict(),
            "best_threshold": best_threshold,
            "in_features": features.shape[1]
        }, MODEL_SAVE_PATH)

print(f"\n🎉 TRAINING COMPLETE!")
print(f"🏆 Best Validation F1-Score: {best_val_f1:.4f} at Decision Threshold tau = {best_threshold:.2f}")
print(f"💾 Saved trained trigger model to: {MODEL_SAVE_PATH}")
