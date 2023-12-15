import json
import pandas as pd
import glob
import os
import sys
import torch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vs_lda.entimate import proposal_infer_only_image
from vs_lda.utils import (
    calc_roc_auc,
    calculate_centroid_and_average_distance,
    calculate_euclid_sum,
    filter_basic_items,
    is_target_category,
)


# from vs_lda.entimate import proposal_infer, topic_model_infer


filepaths = []
with open(
    "D:/M1/fashion/experiments/vs_lda/data/test_coordinates_file_name_new.txt"
) as f:
    for line in f.readlines():
        filepaths.append(line.rstrip("\n").replace("\\", "/"))
# with open(
#     "D:/M1/fashion/experiments/vs_lda/data/item_id_to_attributes.json",
#     encoding="shift-jis",
# ) as f:
#     to_attributes = json.load(f)

with open(
    "D:/M1/fashion/experiments/vs_lda/data/vectors-image-only.json",
    # "D:/M1/fashion/experiments/vs_lda/data/item_id_to_img_path_caption.json",
    encoding="shift-jis",
) as f:
    to_img_path_caption = json.load(f)

print("load jsons")
proposal_score = []
topic_model_score = []
labels = []

# ポジティブなもの
positive_ave = 0
for fp in filepaths:
    labels.append(1)
    json_dict = pd.read_json(fp, encoding="shift-jis")
    parent_dir = os.path.dirname(fp)
    # items = filter_basic_items(json_dict["items"])
    # items = filter(is_target_category, json_dict["items"])
    items = filter_basic_items(json_dict["items"])
    attributes = []
    vectors = []
    for item in items:
        try:
            itemId = str(item["itemId"])
        except Exception as e:
            print(fp, " : ", e)
            continue
        # vectorを推定
        # image_path, _ = to_img_path_caption[itemId]
        # vector = proposal_infer_only_image(image_path)
        vector = to_img_path_caption[itemId]
        vectors.append(torch.tensor(vector))
    ps = calculate_euclid_sum(vectors)
    proposal_score.append(ps.to("cpu"))
    positive_ave += ps

print("positive fin")
positive_ave /= len(filepaths)


negative_ave = 0
# ネガティブなもの
filepaths = glob.glob(
    "D:/M1/fashion/experiments/vs_lda/data/negative_coordinates/**.json"
)
for fp in filepaths:
    labels.append(0)
    json_dict = pd.read_json(fp, encoding="shift-jis")
    parent_dir = os.path.dirname(fp)
    items = []
    attributes = []
    vectors = []
    for item in filter(is_target_category, json_dict["items"]):
        try:
            itemId = str(item["itemId"])
        except Exception as e:
            print(fp, " : ", e)
            continue

        # vectorを推定
        vector = to_img_path_caption[itemId]
        vectors.append(torch.tensor(vector))
    ps = calculate_euclid_sum(vectors)
    proposal_score.append(ps.to("cpu"))
    negative_ave += ps

negative_ave /= len(filepaths)

print(positive_ave, negative_ave)
# t_auc = calc_roc_auc(labels, topic_model_score, "topic_model")
p_auc = calc_roc_auc(labels, proposal_score, "proposal_model_image_only")

# print(t_auc, p_auc, len(labels))
print(p_auc)
