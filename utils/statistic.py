import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc, confusion_matrix


def get_tpr_at_fpr(tpr_lst, fpr_lst, score_lst, fpr_value):
	"""returns true postive rate and threshold given false positive rate value."""
	abs_fpr = np.absolute(fpr_lst - fpr_value)
	idx_min = np.argmin(abs_fpr)
	fpr_value_target = fpr_lst[idx_min]
	idx = np.max(np.where(fpr_lst == fpr_value_target))
	return tpr_lst[idx], score_lst[idx]


def find_best_threshold(y_trues, y_preds):
    # print("Finding best threshold...")
    best_thre = 0.5
    best_metrics = None
    candidate_thres = list(np.unique(np.sort(y_preds)))
    for thre in candidate_thres:
        metrics = cal_metrics(y_trues, y_preds, threshold=thre)
        if best_metrics is None:
            best_metrics = metrics
            best_thre = thre
        elif metrics.get("ACER") < best_metrics.get("ACER"):
            best_metrics = metrics
            best_thre = thre
    # print(f"Best threshold is {best_thre}")
    return best_thre, best_metrics


def cal_metrics(y_trues, y_preds, threshold=0.5):
    metrics = dict()

    fpr, tpr, thresholds = roc_curve(y_trues, y_preds, pos_label=0)
    metrics.update({"AUC": auc(fpr, tpr)})
    metrics.update({"EER": brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)})
    metrics.update({"Thre": float(interp1d(fpr, thresholds)(metrics.get("EER")))})

    if threshold == 'best':
        _, best_metrics = find_best_threshold(y_trues, y_preds)
        return best_metrics

    elif threshold == 'auto':
        threshold = metrics.get("Thre")

    else:
        metrics.update({"Thre": threshold})

    prediction = 1 - (np.array(y_preds) > threshold).astype(int)

    res = confusion_matrix(y_trues, prediction)
    TP, FN = res[0, :]
    FP, TN = res[1, :]
    metrics.update({"ACC": (TP + TN) / len(y_trues)})

    TP_rate = float(TP / (TP + FN))
    TN_rate = float(TN / (TN + FP))

    metrics.update({"TP_Ratio": TP_rate})
    metrics.update({"NumP": TP + FN})
    metrics.update({"TN_Ratio": TN_rate})
    metrics.update({"NumN": TN + FP})
    metrics.update({"APCER": float(FP / (TN + FP))})
    metrics.update({"BPCER": float(FN / (FN + TP))})
    metrics.update({"ACER": (metrics.get("APCER") + metrics.get("BPCER")) / 2})
    
    tpr_01, _ = get_tpr_at_fpr(tpr, fpr, thresholds, 0.01)
    tpr_05, _ = get_tpr_at_fpr(tpr, fpr, thresholds, 0.05)
    metrics.update({"TPR1%": tpr_01})
    metrics.update({"TPR5%": tpr_05})

    return metrics
