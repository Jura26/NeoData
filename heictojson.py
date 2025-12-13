import os
import json

# folder u kojem su tvoje .heic slike
folder = "C:\\ASP\\NeoData\\TRAIN\\negative"   # promijeni po potrebi

for filename in os.listdir(folder):
    if filename.lower().endswith(".heic"):
        base = filename.rsplit(".", 1)[0]
        json_path = os.path.join(folder, base + ".json")

        data = {
            "image": filename,
            "defects": []
        }

        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

print(" JSON datoteke su uspješno kreirane!")