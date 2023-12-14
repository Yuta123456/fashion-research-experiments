import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from vs_lda.models.FashionItemEncoder import FashionItemEncoder
import torch
from PIL import Image
from torchvision import transforms

import numpy as np
import tomotopy as tp

if torch.cuda.is_available():
    device = torch.device("cuda")  # GPUデバイスを取得
else:
    device = torch.device("cpu")  # CPUデバイスを取得

model = FashionItemEncoder().to(device)
model.load_state_dict(
    torch.load(
        "D:/M1/fashion/experiments/vs_lda/models/model_compatibility_2023-11-11.pth"
    )
)

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
model.eval()

mdl = tp.LDAModel.load(
    # D:\M1\fashion\experiments\vs_lda\models\ctm-fashion-clip-T-100-M-10000-B-1000.bin
    # D:\M1\fashion\experiments\vs_lda\models\lda-fashion-clip-T-100-M-10000-B-1000.bin
    "D:/M1/fashion/experiments/vs_lda/models/lda-fashion-clip-T-100-M-10000-B-1000.bin"
)

print("model init")


def proposal_infer(image_path, caption):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("L")
        image = Image.merge("RGB", [image] * 3)
    input_image = transform(image).to(device)
    input_img = torch.unsqueeze(input_image, 0)
    with torch.no_grad():
        pred = model(input_img, [caption])

    return pred


def proposal_infer_batch(images, captions):
    with torch.no_grad():
        pred = model(images, captions)

    return pred


def topic_model_infer(attributes):
    inf_doc = mdl.make_doc(attributes)

    log_prob = mdl.infer(inf_doc, iter=500)[1]

    return log_prob
