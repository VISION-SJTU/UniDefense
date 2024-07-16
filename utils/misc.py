import os
import sys
import time
import torch
import torch.distributed as dist
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def center_print(content, around='*', repeat_around=10):
    num = repeat_around
    s = around
    print(num * s + ' %s ' % content + num * s)
    

def reduce_tensor(t):
    rt = t.clone()
    dist.all_reduce(rt)
    rt /= float(dist.get_world_size())
    return rt


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


class Timer(object):
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


class AccMeter(object):
    def __init__(self):
        self.nums = 0
        self.acc = 0

    def reset(self):
        self.nums = 0
        self.acc = 0

    def update(self, pred, target):
        if pred.shape[-1] == 1:
            pred = (torch.sigmoid(pred.squeeze()) >= 0.5).int()
        else:
            pred = pred.argmax(1)
        self.nums += target.shape[0]
        self.acc += torch.sum(pred == target)

    def mean_acc(self):
        return self.acc / self.nums


class AUCMeter(object):
    def __init__(self):
        self.score = None
        self.true = None

    def reset(self):
        self.score = None
        self.true = None

    def update(self, score, true):
        score = score.cpu().numpy()
        true = true.flatten().cpu().numpy()
        self.score = score if self.score is None else np.concatenate([self.score, score])
        self.true = true if self.true is None else np.concatenate([self.true, true])

    def mean_auc(self):
        return roc_auc_score(self.true, self.score)

    def curve(self, prefix):
        fpr, tpr, thresholds = roc_curve(self.true, self.score, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)
        print(f"# EER: {eer:.4f}(thresh: {thresh:.4f})")
        torch.save([fpr, tpr, thresholds], os.path.join(prefix, "roc_curve.pickle"))


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count