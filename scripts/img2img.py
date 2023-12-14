import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer, AutoModel
import heapq
import matplotlib.pyplot as plt

class ImgScore:
    def __init__(self, img, score):
        self.img = img
        self.score = score
    
    def __lt__(self, other):
        return self.score > other.score

tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v2")

def show_images(scores):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, score in enumerate(scores):
        axes[i].imshow(score.img.permute(1, 2, 0))  # チャンネルの順序を変更
        axes[i].axis('off')
        print(score.score)
    plt.show()

def img2img(
    input_img,
    img_model:nn.Module,
    device, dataloader,
    top=5
    ):
    heap = []
    input_img = torch.unsqueeze(input_img, 0)

    img_t = input_img.to(device)
    target = img_model(img_t)
    with torch.no_grad():
        for (img, _, _, img_path) in dataloader:
            img = torch.unsqueeze(img, 0)
            img_t = img.to(device)
            pred = img_model(img_t)
            distance = torch.norm(pred - target, p=2)
            distance = distance.to('cpu')
            distance = distance.item()
            score = ImgScore(img_path, distance)
            heapq.heappush(heap, score)
            if len(heap) >= top:
                heapq.heappop(heap)
    return heap