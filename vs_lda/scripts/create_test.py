"""
トピックモデルでの互換性との両方を作る。
"""

import json
import random
import pandas as pd
import glob
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from lda_model.preProcessing.preprocessing import preprocessing
from lda_model.util.is_stopword import is_stopword
from lda_model.util.parse_sentence import parse_sentence
from itertools import combinations


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


filepaths = []
with open(
    "D:/M1/fashion/experiments/vs_lda/data/test_coordinates_file_name_new.txt"
) as f:
    for line in f.readlines():
        filepaths.append(line.rstrip("\n").replace("\\", "/"))

cnt = 0
item_pair_cnt = 0
item_id_to_img_path_caption = {}
for fp in filepaths:
    json_dict = pd.read_json(fp, encoding="shift-jis")
    parent_dir = os.path.dirname(fp)
    items = []
    for item in filter(is_target_category, json_dict["items"]):
        itemId = item["itemId"]
        image_path = parent_dir + "/" + str(itemId) + "_m.jpg"
        try:
            caption = preprocessing(item["expressions"][0], debug=False)
            caption = caption.replace(",", "、")
        except Exception as e:
            caption = ""
        item_id_to_img_path_caption[itemId] = [image_path, caption]
    cnt += 1
    if cnt % 1000 == 0:
        print(f"{cnt * 100/ len(filepaths)} 終了")

with open(
    "D:/M1/fashion\experiments/vs_lda/data/item_id_to_img_path_caption.json",
    "w",
    encoding="shift-jis",
) as f:
    json.dump(item_id_to_img_path_caption, f, ensure_ascii=False)
