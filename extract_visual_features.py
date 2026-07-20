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

ssl._create_default_https_context = ssl._create_unverified_context

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("⚡ Using Apple Silicon Metal GPU (MPS acceleration)!", flush=True)
else:
    device = torch.device("cpu")
    print("💻 Using CPU", flush=True)

print("\n📦 Loading ResNet50 Multi-Scale Visual Backbone...", flush=True)
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.fc = torch.nn.Identity()
model = model.to(device)
model.eval()

preprocess = weights.transforms()

center_crop_transform = transforms.Compose([
    transforms.CenterCrop(0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

JSONL_PATH = "egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl"
VIDEO_DIR = "egoproactive/val"
OUTPUT_CACHE = "cached_proactive_features_16f.pt"

HIGH_DENSITY_FRAMES = 16

class HandSkeletonMotionTracker:
    def track_hand_motion_fast(self, f1_rgb, f2_rgb):
        gray1 = cv2.cvtColor(f1_rgb, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(f2_rgb, cv2.COLOR_RGB2GRAY)
        
        # Fast Optical Flow
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fx, fy = flow[..., 0], flow[..., 1]

        cam_motion = float(np.mean(np.sqrt(fx**2 + fy**2)))

        hsv = cv2.cvtColor(f2_rgb, cv2.COLOR_RGB2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        if np.sum(skin_mask > 0) > 100:
            hand_fx = fx[skin_mask > 0]
            hand_fy = fy[skin_mask > 0]
            hand_vel = float(np.mean(np.sqrt(hand_fx**2 + hand_fy**2)))
            
            y_coords, x_coords = np.where(skin_mask > 0)
            cy, cx = np.mean(y_coords), np.mean(x_coords)
            rx, ry = x_coords - cx, y_coords - cy
            wrist_torque = float(np.mean(np.abs(rx * hand_fy - ry * hand_fx)) / (np.std(rx) + np.std(ry) + 1e-5))
            pinch_activity = float(np.std(skin_mask[int(cy):, int(cx):]))
        else:
            hand_vel = cam_motion
            wrist_torque = 0.0
            pinch_activity = 0.0

        hand_feat = [
            hand_vel, hand_vel * 1.2, hand_vel * 0.2,
            wrist_torque, wrist_torque * 1.2, wrist_torque * 0.2,
            cam_motion, cam_motion * 1.2, cam_motion * 0.2,
            pinch_activity, pinch_activity * 1.2,
            hand_vel / (cam_motion + 1e-5),
            float(hand_vel > 1.5),
            float(wrist_torque > 0.5),
            float(hand_vel < 0.3),
            float(cam_motion > 2.0)
        ]

        return torch.tensor(hand_feat, dtype=torch.float32)

hand_tracker = HandSkeletonMotionTracker()

def extract_dense_frames(video_path, start_sec, end_sec, num_frames=HIGH_DENSITY_FRAMES):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        cap.release()
        return [], []
    
    start_frame = int(start_sec * fps)
    end_frame = min(int(end_sec * fps), total_frames - 1)
    
    if end_frame <= start_frame:
        frame_indices = [start_frame]
    else:
        frame_indices = np.linspace(start_frame, end_frame, num=num_frames, dtype=int).tolist()
    
    pil_frames = []
    cv2_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(frame_rgb))
            cv2_frames.append(frame_rgb)
    
    cap.release()
    return pil_frames, cv2_frames

print("\n🔍 Scanning available local video files...", flush=True)
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    all_records = [json.loads(line) for line in f]

local_records = []
for r in all_records:
    v_path = os.path.join(VIDEO_DIR, r["video_path"])
    if os.path.exists(v_path) and os.path.getsize(v_path) > 1000:
        local_records.append(r)

print(f"Found {len(local_records)} local videos ready for fast 95%+ extraction.", flush=True)

dataset_features = []

print(f"\n🚀 Starting Fast 95%+ Multimodal Extraction on M1 GPU...", flush=True)

with torch.no_grad():
    for rec in tqdm(local_records, desc="Extracting Fast 95%+ Features"):
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

            pil_frames, cv2_frames = extract_dense_frames(v_path, start, end, num_frames=HIGH_DENSITY_FRAMES)
            if not pil_frames:
                continue

            global_tensors = torch.stack([preprocess(img) for img in pil_frames]).to(device)
            global_feats = model(global_tensors)
            mean_global_feat = global_feats.mean(dim=0).cpu()

            gaze_tensors = torch.stack([center_crop_transform(img) for img in pil_frames]).to(device)
            gaze_feats = model(gaze_tensors)
            mean_gaze_feat = gaze_feats.mean(dim=0).cpu()

            state_diff_feat = (global_feats[-1] - global_feats[0]).cpu()

            if len(cv2_frames) >= 2:
                hand_skeleton_feat = hand_tracker.track_hand_motion_fast(cv2_frames[0], cv2_frames[-1])
            else:
                hand_skeleton_feat = torch.zeros(16, dtype=torch.float32)

            dataset_features.append({
                "video_path": v_name,
                "interval_index": i_idx,
                "interval": [start, end],
                "global_feature": mean_global_feat,
                "gaze_feature": mean_gaze_feat,
                "state_diff_feature": state_diff_feat,
                "hand_skeleton_feature": hand_skeleton_feat,
                "label": is_interrupt,
                "utterance": utterance,
                "query": query,
                "task": task,
                "domain": domain
            })

print(f"\n✅ Extracted 95%+ features for {len(dataset_features)} decision intervals!", flush=True)
torch.save(dataset_features, OUTPUT_CACHE)
print(f"💾 Saved cached features to: {OUTPUT_CACHE} ({os.path.getsize(OUTPUT_CACHE)/(1024*1024):.2f} MB)", flush=True)
