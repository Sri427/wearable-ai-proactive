import json
from collections import Counter

JSONL_PATH = "egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl"

print("=== 🧐 Inspecting EgoProactive Metadata Schema & Annotations ===")

try:
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    print(f"Total validation samples: {len(samples)}")
    
    # 1. Print keys of a sample
    sample = samples[0]
    print("\n[Sample Metadata Fields]:")
    for key in sample.keys():
        print(f"  - {key}: {type(sample[key]).__name__}")
    
    # 2. Domain & Task distribution
    domains = Counter(s.get("domain", "Unknown") for s in samples)
    print("\n[Top Domains in Dataset]:")
    for dom, count in domains.most_common(8):
        print(f"  - {dom}: {count} videos")

    # 3. Decision intervals statistics
    total_intervals = 0
    silent_count = 0
    interrupt_count = 0
    
    for s in samples:
        answers = s.get("answers", [])
        total_intervals += len(answers)
        for ans in answers:
            if ans == "$silent$":
                silent_count += 1
            elif "$interrupt$" in ans:
                interrupt_count += 1

    print("\n[Decision Interval Statistics]:")
    print(f"  - Total Candidate Decision Intervals: {total_intervals}")
    print(f"  - Silent Intervals ($silent$): {silent_count} ({silent_count/total_intervals*100:.1f}%)")
    print(f"  - Interrupt Intervals ($interrupt$): {interrupt_count} ({interrupt_count/total_intervals*100:.1f}%)")

    # 4. Print sample proactive dialogues
    print("\n[Sample EgoProactive Annotation Entry]:")
    print(f"  Video: {sample.get('video_path')}")
    print(f"  Task: {sample.get('task')}")
    print(f"  User Query: '{sample.get('query')}'")
    print(f"  Duration: {sample.get('duration_in_sec')}s")
    print(f"  Number of Decision Intervals: {len(sample.get('video_intervals', []))}")
    
    print("\n  [First 3 Decision Intervals]:")
    intervals = sample.get('video_intervals', [])
    answers = sample.get('answers', [])
    for idx in range(min(3, len(intervals))):
        print(f"   - Interval {idx+1} {intervals[idx]}: Answer = {answers[idx]}")

except Exception as e:
    print(f"Error inspecting file: {e}")
