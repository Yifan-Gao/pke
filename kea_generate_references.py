# -- coding: utf-8 --
import os
import json
from tqdm import tqdm

data_path = "/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2"

for lang in ["de", 'fr', 'es', 'it']:
# for lang in ["de",]:
    with open(os.path.join(data_path, "percent-2", f"{lang}.train.json")) as f:
        train = [json.loads(line) for line in f]
    with open(os.path.join(data_path, f"{lang}.dev.json")) as f:
        dev = [json.loads(line) for line in f]
    # with open(os.path.join(data_path, f"{lang}.test.json")) as f:
    #     test = [json.loads(line) for line in f]

    data = train+dev

    save_path = os.path.join(data_path, "percent-2", "baseline", 'kea',)
    os.makedirs(save_path, exist_ok=True)
    # f"gold_annotation_{lang}.txt"
    with open(os.path.join(save_path, f"gold_annotation_{lang}.txt"), 'w', encoding="utf-8") as f:
        for i, ex in tqdm(enumerate(data), total=len(data)):
            f.write(f"{lang}.{i} : " + ex['keywords'].replace(";", ",") + "\n")

