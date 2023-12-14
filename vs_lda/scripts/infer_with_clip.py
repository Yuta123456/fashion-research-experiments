import glob
import json
import sys

from fashion_clip.fashion_clip import FashionCLIP
import numpy as np
from sklearn.metrics import *
import pandas as pd
import os

from vs_lda.utils import is_target_category

fclip = FashionCLIP("fashion-clip")


coordinates_file = []
# positive pair
with open(
    "D:/M1/fashion/experiments/vs_lda/data/test_coordinates_file_name_new.txt"
) as f:
    for line in f.readlines():
        coordinates_file.append(line.rstrip("\n").replace("\\", "/"))

# negative pair
negative_pairs_files = glob.glob(
    "D:/M1/fashion/experiments/vs_lda/data/negative_coordinates/**.json"
)

coordinates_file += negative_pairs_files
images = []

for fp in coordinates_file:
    json_dict = pd.read_json(fp, encoding="shift-jis")
    parent_dir = os.path.dirname(fp)
    items = []
    for item in filter(is_target_category, json_dict["items"]):
        itemId = item["itemId"]
        image_path = parent_dir + "/" + str(itemId) + "_m.jpg"
        images.append(image_path)

item_set = set()
new_images = []
for i in images:
    item_id = i.split("/")[-1]
    if item_id in item_set:
        continue
    new_images.append(i)
    item_set.add(item_id)
images = new_images
images = [i.replace("\\", "/") for i in images]
image_embeddings = fclip.encode_images(images, batch_size=128)
image_embeddings = image_embeddings / np.linalg.norm(
    image_embeddings, ord=2, axis=-1, keepdims=True
)
print("image fin")
category = [
    "Vest_top",
    "Hair/alice_band",
    "Leggings/Tights",
    "T-shirt",
    "Sneakers",
    "Sunglasses",
    "Cardigan",
    "Gloves",
    "Underwear_Tights",
    "Hoodie",
    "Other_shoe",
    "Shorts",
    "Jumpsuit/Playsuit",
    "Dress",
    "Trousers",
    "Belt",
    "Socks",
    "Underwear_bottom",
    "Bodysuit",
    "Hat/beanie",
    "Scarf",
    "Jacket",
    "Other_accessories",
    "Bra",
    "Swimwear_bottom",
    "Blazer",
    "Top",
    "Polo shirt",
    "Sweater",
    "Necklace",
    "Pyjama_set",
    "Blouse",
    "Bag",
    "Shirt",
    "Coat",
    "Boots",
    "Skirt",
    "Garment_Set",
    "Bikini_top",
    "Sandals",
    "Dungarees",
    "Earring",
    "Cap/peaked",
    "Ballerinas",
    "Swimsuit",
    "Hat/brim",
]
materials = [
    "Cotton",
    "Polyester",
    "Rayon",
    "Linen",
    "Wool",
    "Silk",
    "Nylon",
    "Polyurethane",
    "Denim",
    "Spandex",
]
colors = [
    "Black",
    "White",
    "Gray",
    "Navy",
    "Blue",
    "Red",
    "Green",
    "Brown",
    "Beige",
    "Pink",
    "Yellow",
    "Orange",
    "Purple",
    "Lavender",
    "Teal",
    "Turquoise",
    "Magenta",
    "Olive",
    "Maroon",
    "Charcoal",
]
created_material_labels = []
created_color_labels = []
for c in category:
    for color in colors:
        created_color_labels.append(f"{c}_{color}")
    for m in materials:
        created_material_labels.append(f"{c}_{m}")
label_material_embeddings = fclip.encode_text(created_material_labels, batch_size=32)
label_material_embeddings = label_material_embeddings / np.linalg.norm(
    label_material_embeddings, ord=2, axis=-1, keepdims=True
)
label_color_embeddings = fclip.encode_text(created_color_labels, batch_size=32)
label_color_embeddings = label_color_embeddings / np.linalg.norm(
    label_color_embeddings, ord=2, axis=-1, keepdims=True
)

predicted_materials = label_material_embeddings.dot(image_embeddings.T)
predicted_materials = [
    created_material_labels[k] for k in np.argmax(predicted_materials, axis=0)
]
print("materials fin")
predicted_colors = label_color_embeddings.dot(image_embeddings.T)
predicted_colors = [
    created_color_labels[k] for k in np.argmax(predicted_colors, axis=0)
]
print("colors fin")
item_id_to_attributes = {}
for coordinates_file, m, c in zip(images, predicted_materials, predicted_colors):
    item_id = coordinates_file.split("/")[-1][:-6]
    item_id_to_attributes[item_id] = [m, c]

# D:\M1\fashion\experiments\vs_lda\data
with open(
    "D:/M1/fashion\experiments/vs_lda/data/item_id_to_attributes.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(item_id_to_attributes, f)
