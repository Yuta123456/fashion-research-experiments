import pandas as pd
import os
from PIL import Image
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from vs_lda.utils import filter_basic_items, is_include_basic_items

filepaths = []
with open(
    "D:/M1/fashion/experiments/vs_lda/data/test_coordinates_file_name_new.txt"
) as f:
    for line in f.readlines():
        filepaths.append(line.rstrip("\n").replace("\\", "/"))
new_filepaths = []
cnt = 0
for fp in filepaths:
    json_dict = pd.read_json(fp, encoding="shift-jis")
    parent_dir = os.path.dirname(fp)
    items = []
    items = json_dict["items"]
    # tops, bottoms shoesを一つずつ含んでいないようならはじく
    if not is_include_basic_items(items):
        continue
    cnt += 1
    if cnt % 100 == 0:
        print(f"{cnt * 100 / len(filepaths)} %")
    new_filepaths.append(fp)
with open(
    "D:/M1/fashion/experiments/vs_lda/data/test_coordinates_file_name_new_2.txt", "w"
) as f:
    f.write("\n".join(new_filepaths))
