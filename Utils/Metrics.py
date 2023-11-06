import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix

'''
The order return is Accuracy, Precision, Recall, F1_Score and Specificity
'''

def EvaluationMetricsMatch(y_pred, y_true):
    if not isinstance(y_pred, np.int32):
        y_pred = y_pred.astype(np.int32)
    if not isinstance(y_true, np.int32):
        y_true = y_true.astype(np.int32)

    Metrics = np.zeros(shape=4, dtype=np.float32)

    Metrics[0] = accuracy_score(y_true=y_true, y_pred=y_pred)
    Metrics[1] = precision_score(y_true=y_true, y_pred=y_pred)
    Metrics[2] = recall_score(y_true=y_true, y_pred=y_pred)
    Metrics[3] = f1_score(y_true=y_true, y_pred=y_pred)

    return np.around(Metrics, decimals=3)


def EvaluationMetricsSequence(y_pred, y_true, y_logits):
    if not isinstance(y_pred, np.int32):
        y_pred = y_pred.astype(np.int32)
    if not isinstance(y_true, np.int32):
        y_true = y_true.astype(np.int32)

    Metrics = np.zeros(shape=6, dtype=np.float32)

    Metrics[0] = accuracy_score(y_true=y_true, y_pred=y_pred)
    Metrics[1] = precision_score(y_true=y_true, y_pred=y_pred)
    Metrics[2] = recall_score(y_true=y_true, y_pred=y_pred)
    Metrics[3] = f1_score(y_true=y_true, y_pred=y_pred)
    Metrics[4] = roc_auc_score(y_true=y_true, y_score=y_logits)
    pre, rec, _ = precision_recall_curve(y_true=y_true, probas_pred=y_logits)
    Metrics[5] = auc(rec, pre)

    return np.around(Metrics, decimals=3)


def EvaluationMetricsToken(y_pred, y_true, y_logits, kmer):
    if not isinstance(y_pred, np.int32):
        y_pred = y_pred.astype(np.int32)
    if not isinstance(y_true, np.int32):
        y_true = y_true.astype(np.int32)

    rowLength = 100 - kmer + 1
    y_pred.shape = (-1, rowLength)
    y_logits.shape = (-1, rowLength)
    y_true.shape = (-1, rowLength)

    logits_pad = np.array([])
    pred_pad = np.array([])
    ture_pad = np.array([])

    i_max, j_max = y_pred.shape

    for i in range(i_max):
        logits = [0] * 100
        pred = [0] * 100
        true = [0] * 100
        for j in range(j_max):
            if y_pred[i][j] == 1:
                for k in range(kmer):
                    pred[j + k] = 1
            if y_true[i][j] == 1:
                for k in range(kmer):
                    true[j + k] = 1
            for k in range(kmer):
                logits[j + k] = y_logits[i][j]
        for j in range(j_max):
            if y_pred[i][j] == 1:
                for k in range(kmer):
                    logits[j + k] = y_logits[i][j]

        logits_pad = np.append(logits_pad, logits)
        pred_pad = np.append(pred_pad, pred)
        ture_pad = np.append(ture_pad, true)

    Metrics = np.zeros(shape=6, dtype=np.float32)

    Metrics[0] = accuracy_score(y_true=ture_pad, y_pred=pred_pad)
    Metrics[1] = precision_score(y_true=ture_pad, y_pred=pred_pad)
    Metrics[2] = recall_score(y_true=ture_pad, y_pred=pred_pad)
    Metrics[3] = f1_score(y_true=ture_pad, y_pred=pred_pad)
    Metrics[4] = roc_auc_score(y_true=ture_pad, y_score=logits_pad)
    pre, rec, _ = precision_recall_curve(y_true=ture_pad, probas_pred=logits_pad)
    Metrics[5] = auc(rec, pre)

    return np.around(Metrics, decimals=3)