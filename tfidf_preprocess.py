# -- coding: utf-8 --
import os
import json
from tqdm import tqdm

data_path = "/home/ec2-user/quic-efs/user/yifangao/multilingual_dataset/full_data/processed/numkps5-30-partially-aligned-from-p2"

for lang in ["de", 'fr', 'es', 'it']:

    with open(os.path.join(data_path, "percent-2", f"{lang}.train.json")) as f:
        train = [json.loads(line) for line in f]
    with open(os.path.join(data_path, f"{lang}.dev.json")) as f:
        dev = [json.loads(line) for line in f]
    with open(os.path.join(data_path, f"{lang}.test.json")) as f:
        test = [json.loads(line) for line in f]

    save_path = os.path.join(data_path, "percent-2", "baseline", 'tfidf', 'docs', f"{lang}",)
    os.makedirs(save_path, exist_ok=True)

    for i, ex in tqdm(enumerate(test), total=len(test)):
        with open(os.path.join(save_path, f"{lang}.{i}.txt"), 'w', encoding="utf-8") as f:
            f.write(ex['content'].replace("<title> ", "").replace("<context> ", "") + "\n")

# save_path = os.path.join(data_path, "percent-2", "baseline", 'tfidf', 'test',)
# os.makedirs(save_path, exist_ok=True)
# with open(os.path.join(save_path, f"{lang}.test.txt"), 'w', encoding="utf-8") as f:
#     for ex in test:
#         f.write(ex['content'].replace("<title> ", "").replace("<context> ", "") + "\n")
