import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

torch.manual_seed(42)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

ENRICHED_CACHE_PATH = "cached_intent_spatial_features_16f.pt"
MODEL_SAVE_PATH = "trigger_classifier_intent_spatial.pt"

# User requested target threshold tau = 0.30 (30%)
TARGET_THRESHOLD = 0.30

print("=== 🧠 Training High-Density Multimodal (16 Snapshots + Gaze + Intent) Classifier ===", flush=True)

data = torch.load(ENRICHED_CACHE_PATH, weights_only=False)
print(f"Loaded {len(data)} high-density 16-frame enriched samples.", flush=True)

global_feats = torch.stack([d["global_feature"] for d in data])     # (N, 2048)
gaze_feats = torch.stack([d["gaze_feature"] for d in data])         # (N, 2048)
diff_feats = torch.stack([d["state_diff_feature"] for d in data])   # (N, 2048)
intent_feats = torch.stack([d["intent_feature"] for d in data])     # (N, 128)
spatial_feats = torch.stack([d["spatial_feature"] for d in data])    # (N, 3)
labels = torch.tensor([d["label"] for d in data], dtype=torch.long) # (N,)

num_samples = len(labels)
num_silent = (labels == 0).sum().item()
num_interrupt = (labels == 1).sum().item()

print(f"Class Balance: Silent=$silent$ ({num_silent}), Interrupt=$interrupt$ ({num_interrupt})")

indices = torch.randperm(num_samples)
train_size = int(0.8 * num_samples)

train_idx, val_idx = indices[:train_size], indices[train_size:]

class MultimodalDenseDataset(Dataset):
    def __init__(self, glob, gaze, diff, intent, sp, y):
        self.glob = glob
        self.gaze = gaze
        self.diff = diff
        self.intent = intent
        self.sp = sp
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.glob[i], self.gaze[i], self.diff[i], self.intent[i], self.sp[i], self.y[i]

train_ds = MultimodalDenseDataset(global_feats[train_idx], gaze_feats[train_idx], diff_feats[train_idx], intent_feats[train_idx], spatial_feats[train_idx], labels[train_idx])
val_ds = MultimodalDenseDataset(global_feats[val_idx], gaze_feats[val_idx], diff_feats[val_idx], intent_feats[val_idx], spatial_feats[val_idx], labels[val_idx])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# 4-Stream Championship Multimodal Fusion Architecture
class ChampionshipMultimodalClassifier(nn.Module):
    def __init__(self, vis_dim=2048, intent_dim=128, spatial_dim=3):
        super().__init__()
        # Stream 1: Global Visual Encoder
        self.glob_encoder = nn.Sequential(
            nn.Linear(vis_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        # Stream 2: Center Gaze Focus Encoder
        self.gaze_encoder = nn.Sequential(
            nn.Linear(vis_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        # Stream 3: Intent Graph Encoder
        self.intent_encoder = nn.Sequential(
            nn.Linear(intent_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        # Stream 4: Motion Velocity & Spatial Progress Encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_dim, 32),
            nn.BatchNorm1d(32),
            nn.GELU()
        )
        # Fusion Layer (256 + 256 + 64 + 32 = 608)
        self.fusion_net = nn.Sequential(
            nn.Linear(608, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, glob, gaze, diff, intent, spatial):
        h_glob = self.glob_encoder(glob)
        h_gaze = self.gaze_encoder(gaze)
        h_intent = self.intent_encoder(intent)
        h_spatial = self.spatial_encoder(spatial)
        
        fused = torch.cat([h_glob, h_gaze, h_intent, h_spatial], dim=-1)
        return self.fusion_net(fused).squeeze(-1)

model = ChampionshipMultimodalClassifier().to(device)

pos_weight = torch.tensor([num_silent / max(1, num_interrupt)]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)

print("\n🔥 Training Multimodal 16-Snapshot Classifier on M1 GPU...", flush=True)

best_val_f1 = 0.0
best_threshold = TARGET_THRESHOLD
epochs = 60

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for b_glob, b_gaze, b_diff, b_intent, b_sp, b_y in train_loader:
        b_glob = b_glob.to(device)
        b_gaze = b_gaze.to(device)
        b_diff = b_diff.to(device)
        b_intent = b_intent.to(device)
        b_sp = b_sp.to(device)
        b_y = b_y.to(device).float()
        
        optimizer.zero_grad()
        logits = model(b_glob, b_gaze, b_diff, b_intent, b_sp)
        loss = criterion(logits, b_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    scheduler.step()

    # Validation at user-requested threshold tau = 0.30
    model.eval()
    val_probs = []
    val_targets = []
    with torch.no_grad():
        for b_glob, b_gaze, b_diff, b_intent, b_sp, b_y in val_loader:
            b_glob = b_glob.to(device)
            b_gaze = b_gaze.to(device)
            b_diff = b_diff.to(device)
            b_intent = b_intent.to(device)
            b_sp = b_sp.to(device)
            probs = torch.sigmoid(model(b_glob, b_gaze, b_diff, b_intent, b_sp)).cpu().numpy()
            val_probs.extend(probs)
            val_targets.extend(b_y.numpy())

    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)

    # Calculate metrics at tau = 0.30
    preds_tau = (val_probs >= TARGET_THRESHOLD).astype(int)
    tp = np.sum((preds_tau == 1) & (val_targets == 1))
    fp = np.sum((preds_tau == 1) & (val_targets == 0))
    fn = np.sum((preds_tau == 0) & (val_targets == 1))

    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    if epoch % 10 == 0 or epoch == epochs:
        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {total_loss/len(train_loader):.4f} | Val F1: {f1:.4f} (Prec: {prec:.4f}, Rec: {rec:.4f} @ tau={TARGET_THRESHOLD:.2f})")

    if f1 > best_val_f1:
        best_val_f1 = f1
        torch.save({
            "model_state": model.state_dict(),
            "best_threshold": TARGET_THRESHOLD
        }, MODEL_SAVE_PATH)

print(f"\n🎉 16-SNAPSHOT MULTIMODAL TRAINING COMPLETE!")
print(f"🏆 Best Validation F1-Score: {best_val_f1:.4f} at Decision Threshold tau = {TARGET_THRESHOLD:.2f}")
print(f"💾 Saved trained model to: {MODEL_SAVE_PATH}")
