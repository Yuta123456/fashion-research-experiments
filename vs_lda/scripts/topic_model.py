import numpy as np
import tomotopy as tp

# モデルを読み込む
# D:/M1/fashion/experiments/vs_lda/models/lda-fashion-clip-T-100-M-10000-B-1000.bin
# D:/M1/fashion/experiments/vs_lda/models/ctm-fashion-clip-T-100-M-10000-B-1000.bin
mdl = tp.CTModel.load(
    "D:/M1/fashion/experiments/vs_lda/models/ctm-fashion-clip-T-100-M-10000-B-1000.bin"
)

fp = "./a.txt"
with open(fp, "r", encoding="utf-8") as f:
    coordinates = f.read().splitlines()


def calc_coordinate_score(attributes):
    inf_doc = mdl.make_doc(attributes)

    log_prob = mdl.infer(inf_doc, iter=500)[1]

    return log_prob
