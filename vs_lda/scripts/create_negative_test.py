import json
import random
import pandas as pd
import glob
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


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

# with open(
#     "D:/M1/fashion/experiments/vs_lda/data/item_id_to_img_path_caption.json",
#     encoding="shift-jis",
# ) as f:
#     to_img_path_caption = json.load(f)


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


def get_category(garment):
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
        return "tops"
        # , "ショートパンツ"入れ忘れた
    if garment_kind in ["スカート", "ロングスカート", "ロングパンツ"]:
        return "bottoms"

    if garment_kind in ["ブーツ", "パンプス", "スニーカー", "靴", "サンダル"]:
        return "shoes"


items = {"tops": [], "bottoms": [], "shoes": []}
for fp in filepaths:
    json_dict = pd.read_json(fp, encoding="shift-jis")
    for item in filter(is_target_category, json_dict["items"]):
        category = get_category(item)
        items[category].append(item)

# print(len(items["tops"]), len(items["bottoms"]), len(items["shoes"]))
# D:/M1/fashion/experiments/vs_lda/data/negative_coordinates


# for i in range(10000):
#     tops = random.sample(items["tops"], 1)[0]
#     bottoms = random.sample(items["bottoms"], 1)[0]
#     shoes = random.sample(items["shoes"], 1)[0]
#     coordinate = {"items": [tops, bottoms, shoes]}
#     with open(
#         f"D:/M1/fashion/experiments/vs_lda/data/negative_coordinates/{random.randint(10000000, 1000000000)}.json",
#         "w",
#     ) as f:
#         json.dump(
#             coordinate,
#             f,
#             ensure_ascii=False,
#         )
#     if i % 100 == 0:
#         print(f"{i / 100}%終わり")
