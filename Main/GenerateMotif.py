import os

from tqdm import tqdm

import numpy as np
import sklearn.model_selection
import torch
import torch.nn as nn
import torch.optim as optim
import transformers.optimization as op
import pandas as pd
from Utils.MotifDiscovery import MotifDiscovery
import Dataset.DataLoader
import Dataset.DataReader
import Model.BertSNR

import Utils.Metrics
import Utils.Threshold


def set_seed(args):
    np.random.seed(args)
    torch.manual_seed(args)


def kmer2seq(kmers):
    """
    Convert kmers to original sequence

    Arguments:
    kmers -- str, kmers separated by space.

    Returns:
    seq -- str, original sequence.

    """
    kmers_list = kmers.split(" ")
    bases = [kmer[0] for kmer in kmers_list[0:-1]]
    bases.append(kmers_list[-1])
    seq = "".join(bases)
    assert len(seq) == len(kmers_list) + len(kmers_list[0]) - 1
    return seq

'''
    Initialization as follows:
'''
KMER = 3
TFsName = 'ZEB1'
motif_len = 11
set_seed(1)
use_gpu = torch.cuda.is_available()
print(use_gpu)
'''
    Initialization End    
'''

'''
    Main Process as follows: 
'''

TestSequence = Dataset.DataReader.DataReaderPrecitBERT('../Dataset/ReMap/' + TFsName + '.bed.txt')
TestSequence = np.array(TestSequence)

ThresholdValue = 0.5

LossFunction = nn.BCELoss().cuda()

BatchSize = 64

NeuralNetwork = Model.BertSNR.BERTSNR(KMER).cuda()

TestLoader = Dataset.DataLoader.SampleLoaderPredictUnlabelBERT(Sequence=TestSequence, BatchSize=BatchSize)

NeuralNetwork.load_state_dict(torch.load('ModelWeight/multiModel/' + TFsName + '/pytorch_model.bin'))
# NeuralNetwork.load_state_dict(torch.load('ModelWeight/multiModel/pytorch_model.bin'))

# valid
NeuralNetwork.eval()
ValidProgressBar = tqdm(TestLoader)
pred = np.array([])

for data in ValidProgressBar:
    X = data
    _, token_logits = NeuralNetwork(X)
    Prediction = Utils.Threshold.Threshold(YPredicted=token_logits.cpu(), ThresholdValue=ThresholdValue)
    pred = np.append(pred, Prediction)

sequence = pd.DataFrame(TestSequence)
label = pred.reshape(-1, 98).astype(np.int64)

sequence_list = []
for row in sequence[0]:
    sequence_list.append(kmer2seq(row))

outfile = open('../Algorithm/DiscoveredMotifs/BertSNR/' + TFsName + '.txt', 'a')

count = 0
for index in range(len(label)):
    row = list(label[index])
    dense_label = [0] * (len(row) + KMER - 1)
    for i in range(len(row)):
        if row[i] == 1:
            for j in range(KMER):
                dense_label[i + j] = 1
    motif = MotifDiscovery(sequence_list[index], dense_label, motif_len)
    if 'N' not in motif:
        count += 1
        outfile.write(motif + '\n')
print(count)

