import os
import sys
import json
import ssl
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.models as models
import torchvision.transforms as transforms

# Bypass SSL verification for model weight downloads
ssl._create_default_https_context = ssl._create_unverified_context

# 1. Device configuration (Apple Silicon Metal MPS if available)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("⚡ Using Apple Silicon Metal GPU (MPS acceleration)!", flush=True)
else:
    device = torch.device("cpu")
    print("💻 Using CPU", flush=True)

# 2. Load high-performance visual backbone
print("\n📦 Loading ResNet50 visual backbone...", flush=True)
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.fc = torch.nn.Identity()
model = model.to(device)
model.eval()

preprocess = weights.transforms()

# Center Gaze Crop Transform (Focus on center 50% of the image where user eyes/gaze look)
center_crop_transform = transforms.Compose([
    transforms.CenterCrop(0.5), # 50% center crop
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

JSONL_PATH = "egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl"
VIDEO_DIR = "egoproactive/val"
OUTPUT_CACHE = "cached_proactive_features_16f.pt"

# High-density frame sampling: 16 snapshots per interval!
HIGH_DENSITY_FRAMES = 16

def extract_dense_frames(video_path, start_sec, end_sec, num_frames=HIGH_DENSITY_FRAMES):
    """Extract `num_frames` high-density PIL frames across interval."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        cap.release()
        return []
    
    start_frame = int(start_sec * fps)
    end_frame = min(int(end_sec * fps), total_frames - 1)
    
    if end_frame <= start_frame:
        frame_indices = [start_frame]
    else:
        frame_indices = np.linspace(start_frame, end_frame, num=num_frames, dtype=int).tolist()
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames

print("\n🔍 Scanning available local video files...", flush=True)
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    all_records = [json.loads(line) for line in f]

local_records = []
for r in all_records:
    v_path = os.path.join(VIDEO_DIR, r["video_path"])
    if os.path.exists(v_path) and os.path.getsize(v_path) > 1000:
        local_records.append(r)

print(f"Found {len(local_records)} local videos ready for 16-snapshot extraction.", flush=True)

dataset_features = []

print(f"\n🚀 Starting High-Density (16 Snapshots + Center Gaze Crop) Extraction on M1 GPU...", flush=True)

with torch.no_grad():
    for rec in tqdm(local_records, desc="Extracting 16-Frame Features"):
        v_name = rec["video_path"]
        v_path = os.path.join(VIDEO_DIR, v_name)
        intervals = rec.get("video_intervals", [])
        answers = rec.get("answers", [])
        query = rec.get("query", "")
        task = rec.get("task", "")
        domain = rec.get("domain", "")

        for i_idx, (start, end) in enumerate(intervals):
            if i_idx >= len(answers):
                continue
            
            raw_ans = answers[i_idx]
            is_interrupt = 1 if "$interrupt$" in raw_ans else 0
            utterance = raw_ans.replace("$interrupt$", "").strip() if is_interrupt else ""

            pil_frames = extract_dense_frames(v_path, start, end, num_frames=HIGH_DENSITY_FRAMES)
            if not pil_frames:
                continue

            # Stream 1: Global Frame Embeddings (16 frames)
            global_tensors = torch.stack([preprocess(img) for img in pil_frames]).to(device)
            global_feats = model(global_tensors) # Shape: (16, 2048)
            mean_global_feat = global_feats.mean(dim=0).cpu()

            # Stream 2: Center Gaze Focus Embeddings (16 center-cropped gaze frames)
            gaze_tensors = torch.stack([center_crop_transform(img) for img in pil_frames]).to(device)
            gaze_feats = model(gaze_tensors) # Shape: (16, 2048)
            mean_gaze_feat = gaze_feats.mean(dim=0).cpu()

            # Stream 3: Before-vs-After Object State Difference Vector
            state_diff_feat = (global_feats[-1] - global_feats[0]).cpu()

            # Stream 4: Motion Velocity & Wrist Rotation Proxy (Frame-to-frame delta)
            if len(global_feats) > 1:
                frame_deltas = torch.norm(global_feats[1:] - global_feats[:-1], dim=1)
                motion_velocity = frame_deltas.mean().item()
            else:
                motion_velocity = 0.0

            dataset_features.append({
                "video_path": v_name,
                "interval_index": i_idx,
                "interval": [start, end],
                "global_feature": mean_global_feat,         # 2048-dim
                "gaze_feature": mean_gaze_feat,             # 2048-dim
                "state_diff_feature": state_diff_feat,      # 2048-dim
                "motion_velocity": motion_velocity,         # Float
                "label": is_interrupt,
                "utterance": utterance,
                "query": query,
                "task": task,
                "domain": domain
            })

print(f"\n✅ Extracted 16-snapshot + gaze features for {len(dataset_features)} decision intervals!", flush=True)
torch.save(dataset_features, OUTPUT_CACHE)
print(f"💾 Saved cached features to: {OUTPUT_CACHE} ({os.path.getsize(OUTPUT_CACHE)/(1024*1024):.2f} MB)", flush=True)
