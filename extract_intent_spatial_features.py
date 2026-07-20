import os
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

CACHE_PATH = "cached_proactive_features_16f.pt"
ENRICHED_CACHE_PATH = "cached_intent_spatial_features_16f.pt"

print("=== 🧠 Building Multimodal 16-Snapshot + Gaze + Intent Features ===", flush=True)

data = torch.load(CACHE_PATH, weights_only=False)
print(f"Loaded {len(data)} high-density samples.", flush=True)

# TF-IDF Intent Vectorizer
all_texts = [f"{s['query']} {s['task']} {s['domain']}" for s in data]
vectorizer = TfidfVectorizer(max_features=128)
text_intents = vectorizer.fit_transform(all_texts).toarray()

# Group samples by video_path
video_groups = {}
for idx, sample in enumerate(data):
    v_path = sample["video_path"]
    if v_path not in video_groups:
        video_groups[v_path] = []
    video_groups[v_path].append((idx, sample))

enriched_dataset = []

for v_path, items in video_groups.items():
    items.sort(key=lambda x: x[1]["interval"][0])
    total_intervals = len(items)
    
    for i, (orig_idx, sample) in enumerate(items):
        global_feat = sample["global_feature"]          # 2048-dim
        gaze_feat = sample["gaze_feature"]              # 2048-dim
        diff_feat = sample["state_diff_feature"]        # 2048-dim
        intent_feat = torch.tensor(text_intents[orig_idx], dtype=torch.float32) # 128-dim
        
        motion_vel = sample["motion_velocity"]
        progress = float(i) / float(max(1, total_intervals - 1))
        duration = float(sample["interval"][1] - sample["interval"][0])
        
        spatial_vec = torch.tensor([motion_vel, progress, duration], dtype=torch.float32) # 3-dim
        
        enriched_dataset.append({
            "video_path": v_path,
            "interval_index": sample["interval_index"],
            "interval": sample["interval"],
            "global_feature": global_feat,
            "gaze_feature": gaze_feat,
            "state_diff_feature": diff_feat,
            "intent_feature": intent_feat,
            "spatial_feature": spatial_vec,
            "label": sample["label"],
            "utterance": sample["utterance"],
            "query": sample["query"],
            "task": sample["task"],
            "domain": sample["domain"]
        })

torch.save(enriched_dataset, ENRICHED_CACHE_PATH)
print(f"✅ Enriched {len(enriched_dataset)} high-density samples!")
print(f"💾 Saved to: {ENRICHED_CACHE_PATH} ({os.path.getsize(ENRICHED_CACHE_PATH)/(1024*1024):.2f} MB)", flush=True)
