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
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vs_lda.entimate import proposal_infer_batch, proposal_infer_only_images_batch

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
# captions = []
images = []
vectors = {}
BATCH_SIZE = 10
cnt = 0
targets = list(to_img_path_caption.items())
for key, v in targets:
    ids.append(key)
    cnt += 1
    if cnt % 1000 == 0:
        print(f"{cnt * 100 / len(targets)} %")
    images.append(to_image(v[0]))
images = torch.stack(images)
print(len(images), len(ids))
for i in range(0, len(ids), BATCH_SIZE):
    if i % 10 * BATCH_SIZE == 0:
        print(f"{i * 100 / len(ids)} %")
    s = i
    e = min(i + BATCH_SIZE, len(ids))

    pred = proposal_infer_only_images_batch(images[s:e].to(torch.device("cuda")))
    pred_ids = ids[s:e]
    for j in range(len(pred)):
        itemId = pred_ids[j]
        vectors[itemId] = pred[j].to("cpu").tolist()

with open("D:/M1/fashion/experiments/vs_lda/data/vectors-image-only.json", "w") as f:
    json.dump(vectors, f)
