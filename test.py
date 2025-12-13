import os, json
from collections import Counter

NEG_DIR = r"c:\Users\Jura Slibar\Desktop\NeoData\TRAIN\negative"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

json_files = [os.path.join(NEG_DIR, f) for f in os.listdir(NEG_DIR) if f.lower().endswith(".json")]
print("JSON files:", len(json_files))

all_top_keys = Counter()
type_counts = Counter()

sample = None
for p in json_files:
    try:
        data = load_json(p)
    except Exception as e:
        print("Failed:", p, e)
        continue

    if isinstance(data, dict):
        all_top_keys.update(data.keys())

        # common guesses for type fields
        for k in ("type", "element_type", "elementType", "product", "variant", "model"):
            if k in data:
                type_counts[str(data[k])] += 1
                break

        if sample is None:
            sample = (p, data)

print("\nTop-level keys (most common):")
for k, c in all_top_keys.most_common(30):
    print(f"  {k}: {c}")

print("\nType counts (if found):")
for t, c in type_counts.most_common():
    print(f"  {t}: {c}")

if sample:
    p, data = sample
    print("\nSample file:", os.path.basename(p))
    print("Sample keys:", list(data.keys())[:50])