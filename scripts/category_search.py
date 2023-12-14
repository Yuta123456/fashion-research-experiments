import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import heapq
import matplotlib.pyplot as plt
from PIL import Image
from experiments.img2img import ImgScore
def get_image_category(img_path):
    coordinate_info = get_coordinate_info(img_path)
    fashion_item_id = img_path.split('\\')[-1]
    # itemsの中のidに一致するものを探す
    for item in coordinate_info['items']:
        if item['itemId'] == int(fashion_item_id):
            return item['category x color'] 

    
def get_coordinate_info(img_path: str):
    # D:\M1\fashion\IQON\IQON3000\77\3907846\12727752_m.jpg
    coordinate_id = img_path.split('\\')[-2]
    coordinate_json_path = img_path.split('\\')[:-1].join('\\') + coordinate_id + '_new.json'
    try:
        d = open(coordinate_json_path, 'r')
        json_data = pd.read_json(d)
    except Exception as e:
        print(e)
    return json_data


def image_search_with_category(
    target_img_path,
    img_model:nn.Module,
    device, dataloader,
    top=10
    ):
    heap = []
    image = Image.open(target_img_path)
    if image.mode != 'RGB':
        image = image.convert('L')
        image = Image.merge('RGB', [image] * 3)
    input_image = img_model.transform(image)
    input_img = torch.unsqueeze(input_image, 0)

    target_img_category = get_image_category(target_img_path)
    img_t = input_img.to(device)
    target = img_model(img_t)
    with torch.no_grad():
        for (img, _, _, img_path) in dataloader:
            category = get_image_category(img_path)
            if (target_img_category != category):
                continue
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