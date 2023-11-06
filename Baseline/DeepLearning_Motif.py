import numpy as np
import sklearn.model_selection
import pandas as pd
import torch

import Dataset.DataLoader
import Dataset.DataReader
import Model.DeepSNR
import Model.D_AEDNet
import Utils.Metrics
import Utils.Threshold
import os
from Utils.MotifDiscovery import MotifDiscovery

'''
    Initialization as follows:
'''
# NeuralNetworkName = 'DeepSNR'
NeuralNetworkName = 'D_AEDNet'
'''
    Initialization End    
'''
TFsName = 'PAX5'
motif_len = 17
print(TFsName)
data_path = '../Dataset/ReMap/' + TFsName + '.bed.txt'

FeatureMatrix = Dataset.DataReader.DataReaderPrecit(path=data_path)
FeatureMatrix = np.array(FeatureMatrix)

ThresholdValue = 0.5

if NeuralNetworkName == 'DeepSNR':
    NeuralNetwork = Model.DeepSNR.DeepSNR(SequenceLength=100, MotifLength=15)
else:
    NeuralNetwork = Model.D_AEDNet.D_AEDNN(SequenceLength=100)

TestFeatureMatrix = torch.tensor(FeatureMatrix, dtype=torch.float32).unsqueeze(dim=1)
TestLoader = Dataset.DataLoader.SampleLoaderPredict(FeatureMatrix=TestFeatureMatrix, BatchSize=64)

NeuralNetwork.load_state_dict(
    torch.load('Weight/' + NeuralNetworkName + '/' + TFsName + '_1.pth'))

NeuralNetwork.eval()
pred = np.array([])
for step, data in enumerate(TestLoader, start=0):
    X = data
    Logits = NeuralNetwork(X)
    Prediction = Utils.Threshold.Threshold(YPredicted=Logits, ThresholdValue=ThresholdValue)
    pred = np.append(pred, Prediction)

data_sequence = pd.read_csv(data_path, header=None)

sequence = data_sequence[0].tolist()
label = pred.reshape(-1, 100).astype(np.int64)

outfile = open('../Algorithm/DiscoveredMotifs/' + NeuralNetworkName + '/' + TFsName + '.txt', 'a')
# outfile.write('>seq' + str(1) + '\n')

count = 0
for index in range(len(sequence)):
    motif = MotifDiscovery(sequence[index], label[index], motif_len)
    if 'N' not in motif:
        # outfile.write('>seq' + str(index) + '\n')
        count += 1
        outfile.write(motif + '\n')
    if count >= 10000:
        break

#
# for tf in os.listdir('../Dataset/ReMap')[6:]:
#     TFsName = tf[:-8]
#     print(TFsName)
#     data_path = '../Dataset/ReMap/' + TFsName + '.bed.txt'
#
#     FeatureMatrix = Dataset.DataReader.DataReaderPrecit(path=data_path)
#     FeatureMatrix = np.array(FeatureMatrix)
#
#     ThresholdValue = 0.5
#
#     if NeuralNetworkName == 'DeepSNR':
#         NeuralNetwork = Model.DeepSNR.DeepSNR(SequenceLength=100, MotifLength=15)
#     else:
#         NeuralNetwork = Model.D_AEDNet.D_AEDNN(SequenceLength=100)
#
#     TestFeatureMatrix = torch.tensor(FeatureMatrix, dtype=torch.float32).unsqueeze(dim=1)
#     TestLoader = Dataset.DataLoader.SampleLoaderPredict(FeatureMatrix=TestFeatureMatrix, BatchSize=64)
#
#     NeuralNetwork.load_state_dict(
#         torch.load('Weight/' + NeuralNetworkName + '/' + TFsName + '_1.pth'))
#
#     NeuralNetwork.eval()
#     pred = np.array([])
#     for step, data in enumerate(TestLoader, start=0):
#         X = data
#         Logits = NeuralNetwork(X)
#         Prediction = Utils.Threshold.Threshold(YPredicted=Logits, ThresholdValue=ThresholdValue)
#         pred = np.append(pred, Prediction)
#
#     data_sequence = pd.read_csv(data_path, header=None)
#
#     sequence = data_sequence[0].tolist()
#     label = pred.reshape(-1, 100).astype(np.int64)
#
#     outfile = open('../Algorithm/DiscoveredMotifs/' + NeuralNetworkName + '/' + TFsName + '.txt', 'a')
#     # outfile.write('>seq' + str(1) + '\n')
#
#     count = 0
#     for index in range(len(sequence)):
#         motif = MotifDiscovery(sequence[index], label[index], 11)
#         if 'N' not in motif:
#             # outfile.write('>seq' + str(index) + '\n')
#             count += 1
#             outfile.write(motif + '\n')
#         if count >= 10000:
#             break
#
