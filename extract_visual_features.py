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

# 2. Load high-performance visual backbone (ResNet50 / ConvNeXt)
print("\n📦 Loading visual backbone (ResNet50 feature extractor)...", flush=True)
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.fc = torch.nn.Identity() # Remove final classifier head to output 2048-dim embedding
model = model.to(device)
model.eval()

# Preprocessing transforms matching ImageNet standard
preprocess = weights.transforms()

JSONL_PATH = "egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl"
VIDEO_DIR = "egoproactive/val"
OUTPUT_CACHE = "cached_proactive_features.pt"

def extract_frames_for_interval(video_path, start_sec, end_sec, num_frames=4):
    """Extract `num_frames` uniformly spaced PIL Image frames for a given time interval."""
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

print("\n🔍 Scanning available video files...", flush=True)
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    all_records = [json.loads(line) for line in f]

local_records = []
for r in all_records:
    v_path = os.path.join(VIDEO_DIR, r["video_path"])
    if os.path.exists(v_path) and os.path.getsize(v_path) > 1000:
        local_records.append(r)

print(f"Found {len(local_records)} local videos ready for feature extraction.", flush=True)

dataset_features = []

print("\n🚀 Starting visual feature extraction on M1 GPU...", flush=True)
with torch.no_grad():
    for rec in tqdm(local_records, desc="Extracting Features"):
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
            # Binary label: 0 for $silent$, 1 for $interrupt$
            is_interrupt = 1 if "$interrupt$" in raw_ans else 0
            utterance = raw_ans.replace("$interrupt$", "").strip() if is_interrupt else ""

            pil_frames = extract_frames_for_interval(v_path, start, end, num_frames=4)
            if not pil_frames:
                continue

            # Transform images into tensor batch
            tensors = torch.stack([preprocess(img) for img in pil_frames]).to(device)
            # Pass through ResNet50 -> shape (num_frames, 2048)
            frame_feats = model(tensors)
            # Mean pool across frames -> shape (2048,)
            interval_feat = frame_feats.mean(dim=0).cpu()

            dataset_features.append({
                "video_path": v_name,
                "interval_index": i_idx,
                "interval": [start, end],
                "feature": interval_feat,
                "label": is_interrupt,
                "utterance": utterance,
                "query": query,
                "task": task,
                "domain": domain
            })

print(f"\n✅ Extracted visual features for {len(dataset_features)} decision intervals across {len(local_records)} videos!", flush=True)
torch.save(dataset_features, OUTPUT_CACHE)
print(f"💾 Saved cached features to: {OUTPUT_CACHE} ({os.path.getsize(OUTPUT_CACHE)/(1024*1024):.2f} MB)", flush=True)
