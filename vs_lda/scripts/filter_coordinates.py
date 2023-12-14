import pandas as pd
import os
from PIL import Image

filepaths = []
with open("D:/M1/fashion/experiments/vs_lda/data/test_coordinates_file_name.txt") as f:
    for line in f.readlines():
        filepaths.append(line.rstrip("\n").replace("\\", "/"))
new_filepaths = []
cnt = 0
for fp in filepaths:
    json_dict = pd.read_json(fp, encoding="shift-jis")
    parent_dir = os.path.dirname(fp)
    items = []
    not_open = False
    for item in json_dict["items"]:
        itemId = item["itemId"]
        image_path = parent_dir + "/" + str(itemId) + "_m.jpg"
        try:
            image = Image.open(image_path)
        except Exception as e:
            not_open = True
            break
    if not_open:
        continue
    cnt += 1
    if cnt % 10 == 0:
        print(f"{cnt * 100 / len(filepaths)} %")
    new_filepaths.append(fp)
with open(
    "D:/M1/fashion/experiments/vs_lda/data/test_coordinates_file_name_new.txt", "w"
) as f:
    f.write("\n".join(new_filepaths))
