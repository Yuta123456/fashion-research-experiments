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

def cap2img(text:str,
    cap_model:nn.Module,
    img_model:nn.Module,
    device, dataloader,
    top=5):

    ids = tokenizer.encode(text, return_tensors='pt')
    ids = ids.to(device)
    target = cap_model(ids)
    heap = []
    with torch.no_grad():
        for (img, _, _) in dataloader:
            img_t = img.to(device)
            pred = img_model(img_t)
            distance = torch.norm(pred - target, p=2)
            score = ImgScore(img_t, distance)
            heapq.heappush(heap, score)
            if len(heap) >= top:
                heapq.heappop(heap)
