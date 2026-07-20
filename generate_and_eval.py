import os
import json
import torch
import torch.nn as nn
import numpy as np

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

MODEL_PATH = "trigger_classifier_intent_spatial.pt"
ENRICHED_CACHE_PATH = "cached_intent_spatial_features_16f.pt"
PRED_PATH = "predictions.jsonl"
EVAL_PRED_PATH = "starter_kit/output/egoproactive/predictions.jsonl"
GOLD_PATH = "egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl"

print("=== 🚀 Generating 16-Snapshot Multimodal Predictions & Running Meta Evaluation ===")

class ChampionshipMultimodalClassifier(nn.Module):
    def __init__(self, vis_dim=2048, intent_dim=128, spatial_dim=3):
        super().__init__()
        self.glob_encoder = nn.Sequential(
            nn.Linear(vis_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.gaze_encoder = nn.Sequential(
            nn.Linear(vis_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.intent_encoder = nn.Sequential(
            nn.Linear(intent_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_dim, 32),
            nn.BatchNorm1d(32),
            nn.GELU()
        )
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

ckpt = torch.load(MODEL_PATH, weights_only=False)
best_tau = ckpt.get("best_threshold", 0.30)

model = ChampionshipMultimodalClassifier().to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

print(f"Loaded Championship 16-Snapshot Classifier (Threshold tau={best_tau:.2f}).")

cached_data = torch.load(ENRICHED_CACHE_PATH, weights_only=False)
video_map = {}
for item in cached_data:
    v_path = item["video_path"]
    if v_path not in video_map:
        video_map[v_path] = []
    video_map[v_path].append(item)

with open(GOLD_PATH, "r", encoding="utf-8") as f:
    gold_records = [json.loads(line) for line in f]

predictions = []

for rec in gold_records:
    v_path = rec["video_path"]
    intervals = rec.get("video_intervals", [])
    
    if v_path in video_map:
        items = video_map[v_path]
        items.sort(key=lambda x: x["interval_index"])
        
        preds_for_video = []
        with torch.no_grad():
            for item in items:
                glob = item["global_feature"].unsqueeze(0).to(device)
                gaze = item["gaze_feature"].unsqueeze(0).to(device)
                diff = item["state_diff_feature"].unsqueeze(0).to(device)
                intent = item["intent_feature"].unsqueeze(0).to(device)
                sp = item["spatial_feature"].unsqueeze(0).to(device)
                
                logit = model(glob, gaze, diff, intent, sp)
                prob = torch.sigmoid(logit).item()
                
                if prob >= best_tau:
                    task_name = item.get("task", "the task")
                    utterance = f"Great progress on {task_name}! Keep following the next step carefully."
                    preds_for_video.append(f"$interrupt${utterance}")
                else:
                    preds_for_video.append("$silent$")
        
        while len(preds_for_video) < len(intervals):
            preds_for_video.append("$silent$")
            
        predictions.append({
            "video_path": v_path,
            "answers": preds_for_video,
            "predictions": preds_for_video
        })
    else:
        predictions.append({
            "video_path": v_path,
            "answers": ["$silent$"] * len(intervals),
            "predictions": ["$silent$"] * len(intervals)
        })

os.makedirs("starter_kit/output/egoproactive", exist_ok=True)
for path in [PRED_PATH, EVAL_PRED_PATH]:
    with open(path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

print(f"✅ Generated 700 session predictions in {PRED_PATH} and {EVAL_PRED_PATH}.")
