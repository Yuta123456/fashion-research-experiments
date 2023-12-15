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


def calculate_euclid_sum(tensor_list):
    distance12 = torch.norm(tensor_list[0] - tensor_list[1])
    distance13 = torch.norm(tensor_list[0] - tensor_list[2])
    distance23 = torch.norm(tensor_list[1] - tensor_list[2])

    # ユークリッド距離の合計を計算
    total_distance = distance12 + distance13 + distance23

    return total_distance


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


def is_include_basic_items(items):
    flag = 0
    for garment in items:
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
            flag |= 1
            # , "ショートパンツ"入れ忘れた
        if garment_kind in ["スカート", "ロングスカート", "ロングパンツ"]:
            flag |= 1 << 1

        if garment_kind in ["ブーツ", "パンプス", "スニーカー", "靴", "サンダル"]:
            flag |= 1 << 2
    return flag == 7


def filter_basic_items(items):
    flag = 0
    new_items = []
    for garment in items:
        garment_kind = garment["category x color"].split(" × ")[0]
        # print(garment_kind)
        if (
            garment_kind
            in [
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
            ]
            and (flag & 1) == 0
        ):
            flag |= 1
            new_items.append(garment)
            # , "ショートパンツ"入れ忘れた
        elif garment_kind in ["スカート", "ロングスカート", "ロングパンツ"] and (flag & (1 << 1)) == 0:
            flag |= 1 << 1
            new_items.append(garment)
        elif garment_kind in ["ブーツ", "パンプス", "スニーカー", "靴", "サンダル"] and (
            flag & (1 << 2) == 0
        ):
            flag |= 1 << 2
            new_items.append(garment)
        # print(flag, garment, new_items)
    return new_items
