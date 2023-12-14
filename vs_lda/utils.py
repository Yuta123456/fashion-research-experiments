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
