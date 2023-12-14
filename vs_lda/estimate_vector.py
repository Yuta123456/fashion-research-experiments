"""
batch単位でproposal_modelのvectorを予想するやつ。
output: {
 "itemId": vector
}
"""

import json
from torchvision import transforms
import torch
import os
import sys
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vs_lda.entimate import proposal_infer_batch

# from vs_lda.entimate import proposal_infer, proposal_infer_batch
from PIL import Image

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def to_image(image_path):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("L")
        image = Image.merge("RGB", [image] * 3)

    image = transform(image)
    return image


with open(
    "D:/M1/fashion/experiments/vs_lda/data/item_id_to_img_path_caption.json",
    encoding="shift-jis",
) as f:
    to_img_path_caption = json.load(f)
ids = []
captions = []
images = []
vectors = {}
BATCH_SIZE = 10
for key, v in list(to_img_path_caption.items()):
    ids.append(key)
    captions.append(v[1])
    images.append(to_image(v[0]))
images = torch.stack(images).to(torch.device("cuda"))
for i in range(0, len(ids), BATCH_SIZE):
    s = i
    e = min(i + BATCH_SIZE, len(ids))
    pred = proposal_infer_batch(images[s:e], captions[s:e])
    for j in range(s, e):
        itemId = ids[j]
        vectors[itemId] = pred[j].tolist()

with open("D:/M1/fashion/experiments/vs_lda/data/vectors.json", "w") as f:
    json.dump(vectors, f)
