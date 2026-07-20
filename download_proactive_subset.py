import os
import sys
import json
from huggingface_hub import hf_hub_download

# Read token from environment variable or fallback
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "facebook/wearable-ai"
SUBSET_COUNT = 50

print("=== 🚀 Starting Download of 50-Video Mini-Subset ===", flush=True)

# 1. Read first 50 video filenames from JSONL
jsonl_path = "egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl"
video_paths = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        v_path = data.get("video_path")
        if v_path:
            video_paths.append(v_path)

subset_videos = video_paths[:SUBSET_COUNT]
print(f"Targeting {len(subset_videos)} videos for local subset (~1.5–2 GB)...", flush=True)

os.makedirs("egoproactive/val", exist_ok=True)

downloaded_count = 0
total_bytes = 0

for idx, v_name in enumerate(subset_videos, 1):
    repo_filename = f"egoproactive/val/{v_name}"
    local_target = os.path.join("egoproactive", "val", v_name)
    
    if os.path.exists(local_target) and os.path.getsize(local_target) > 1000:
        file_size = os.path.getsize(local_target)
        total_bytes += file_size
        downloaded_count += 1
        print(f"[{idx}/{len(subset_videos)}] Already downloaded {v_name} ({file_size / (1024*1024):.1f} MB)", flush=True)
        continue

    print(f"[{idx}/{len(subset_videos)}] Downloading {v_name}...", flush=True)
    try:
        file_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=repo_filename,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir="."
        )
        file_size = os.path.getsize(file_path)
        total_bytes += file_size
        downloaded_count += 1
        print(f"   -> Success! ({file_size / (1024*1024):.1f} MB)", flush=True)
    except Exception as e:
        print(f"   -> Failed {v_name}: {e}", flush=True)

print(f"\n🎉 DONE! {downloaded_count}/{SUBSET_COUNT} videos ready (~{total_bytes / (1024*1024*1024):.2f} GB total).", flush=True)
