from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def save_roc(name: str):
    plt.plot(fpr, tpr, marker="o")
    plt.xlabel("FPR: False positive rate")
    plt.ylabel("TPR: True positive rate")
    plt.grid()
    plt.savefig(f"D:/M1/fashion/experiments/vs_lda/result/{name}_roc.png")


y_true = [0, 0, 0, 0, 1, 1, 1, 1]
y_score = [0.2, 0.3, 0.6, 0.8, 0.4, 0.5, 0.7, 101.1]

fpr, tpr, thresholds = roc_curve(y_true, y_score)

auc = roc_auc_score(y_true, y_score)

print(auc)
