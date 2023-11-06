from tqdm import tqdm

import numpy as np
import sklearn.model_selection
import torch
import torch.nn as nn
import torch.optim as optim
import transformers.optimization as op
import pandas as pd

import Dataset.DataLoader
import Dataset.DataReader
import Model.BertSNR

import Utils.Metrics
import Utils.Threshold


def set_seed(args):
    np.random.seed(args)
    torch.manual_seed(args)
'''
    Initialization as follows:
'''
TFsName = 'BACH2'
DataSetName = 'ChIP-seq'
KMER = 3
set_seed(1)
use_gpu = torch.cuda.is_available()
print(use_gpu)
'''
    Initialization End    
'''

'''
    Main Process as follows: 
'''


_, _, _, TestSequence, TestDenseLabel, TestLabel = \
    Dataset.DataReader.DataReaderBERT(TFsName, DataSetName, KMER)
TestSequence, TestDenseLabel, TestLabel = np.array(TestSequence), np.array(TestDenseLabel), np.array(TestLabel)

ThresholdValue = 0.5

LossFunction = nn.BCELoss().cuda()

BatchSize = 64

NeuralNetwork = Model.BertSNR.BERTSNR(KMER).cuda()

TestDenseLabels = torch.tensor(TestDenseLabel)
TestLabels = torch.tensor(TestLabel)
TestLoader = Dataset.DataLoader.SampleLoaderPredictBERT(Sequence=TestSequence, DenseLabel=TestDenseLabel,
                                                  Label=TestLabel, BatchSize=BatchSize)

NeuralNetwork.load_state_dict(torch.load('ModelWeight/multiModel/' + TFsName + '/pytorch_model.bin'))
# NeuralNetwork.load_state_dict(torch.load('ModelWeight/multiModel/pytorch_model.bin'))

# valid
NeuralNetwork.eval()
ValidProgressBar = tqdm(TestLoader)
pred = np.array([])
label = np.array([])
logits = np.array([])
for data in ValidProgressBar:
    X, Y1, _ = data
    X = X
    Y1 = Y1.cuda()
    _, Logits = NeuralNetwork(X)
    Prediction = Utils.Threshold.Threshold(YPredicted=Logits.cpu(), ThresholdValue=ThresholdValue)
    logits = np.append(logits, Logits.cpu().detach().numpy())
    pred = np.append(pred, Prediction)
    label = np.append(label, Y1.cpu())
Performance = np.zeros(shape=6, dtype=np.float32)
Performance[0], Performance[1], Performance[2], Performance[3], Performance[4], Performance[5] \
    = Utils.Metrics.EvaluationMetricsToken(y_pred=pred, y_true=label, y_logits=logits, kmer=KMER)
print('Acc=%.3f, Pre=%.3f, Rec=%.3f, F1-S=%.3f, AUC=%.3f, AUPR=%.3f,' % (
    Performance[0], Performance[1], Performance[2], Performance[3], Performance[4], Performance[5]))

