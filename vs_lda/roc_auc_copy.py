import json
import random
import pandas as pd
import glob
import os
import sys
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vs_lda.entimate import proposal_infer, topic_model_infer


def is_target_category(garment):
    garment_kind = garment["category x color"].split(" × ")[0]
    if garment_kind in [
        "ジャケット",
        "トップス",
        "コート",
        "ニット",
        "タンクトップ",
        "ブラウス",
        "Tシャツ",
        "カーディガン",
        "ダウンジャケット",
        "パーカー",
    ]:
        return True
        # , "ショートパンツ"入れ忘れた
    if garment_kind in ["スカート", "ロングスカート", "ロングパンツ"]:
        return True

    if garment_kind in ["ブーツ", "パンプス", "スニーカー", "靴", "サンダル"]:
        return True

    return False


def calculate_centroid_and_average_distance(tensor_list):
    # 3つのテンソルの重心ベクトルを計算
    centroid = sum(tensor_list) / len(tensor_list)

    # 重心ベクトルに対する各ベクトルのユークリッド距離を計算し合計
    total_distance = 0.0
    for tensor in tensor_list:
        distance = torch.norm(tensor - centroid)  # ユークリッド距離の計算
        total_distance += distance

    # 平均距離を計算
    average_distance = total_distance / len(tensor_list)

    return average_distance


def calc_roc_auc(label, score, name):
    fpr, tpr, _ = roc_curve(label, score)

    auc = roc_auc_score(label, score)
    plt.clf()
    plt.plot(fpr, tpr, marker="o")
    plt.xlabel("FPR: False positive rate")
    plt.ylabel("TPR: True positive rate")
    plt.grid()
    plt.savefig(f"D:/M1/fashion/experiments/vs_lda/result/{name}_roc.png")
    return auc


def calc_filepath_score(filepaths):
    scores = []
    for fp in filepaths:
        json_dict = pd.read_json(fp, encoding="shift-jis")
        vectors = []
        try:
            for item in filter(is_target_category, json_dict["items"]):
                itemId = str(item["itemId"])
                # vectorを推定
                img_path, caption = to_img_path_caption[itemId]
                if caption == "":
                    raise Exception("no caption item include")
                vectors.append(proposal_infer(img_path, caption))
            ps = calculate_centroid_and_average_distance(vectors)
            scores.append(ps.to("cpu"))
        except Exception as e:
            print(e)
            continue
    return scores


filepaths = []
with open(
    "D:/M1/fashion/experiments/vs_lda/data/test_coordinates_file_name_new.txt"
) as f:
    for line in f.readlines():
        filepaths.append(line.rstrip("\n").replace("\\", "/"))
with open(
    "D:/M1/fashion/experiments/vs_lda/data/item_id_to_attributes.json",
    encoding="shift-jis",
) as f:
    to_attributes = json.load(f)

with open(
    "D:/M1/fashion/experiments/vs_lda/data/item_id_to_img_path_caption.json",
    encoding="shift-jis",
) as f:
    to_img_path_caption = json.load(f)

positive_score = calc_filepath_score(filepaths)
labels = [1 for _ in positive_score]

# ネガティブなもの
filepaths = glob.glob(
    "D:/M1/fashion/experiments/vs_lda/data/negative_coordinates/**.json"
)
negative_score = calc_filepath_score(filepaths)
labels += [0 for _ in negative_score]

p_auc = calc_roc_auc(labels, positive_score + negative_score, "proposal_model")

print(p_auc, len(labels))
