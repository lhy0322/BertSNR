import numpy as np
import sklearn.model_selection
import torch

import Dataset.DataLoader
import Dataset.DataReader
import Model.DeepSNR
import Model.D_AEDNet
import Utils.Metrics
import Utils.Threshold

'''
    Initialization as follows:
'''
TFsName = 'CTCF'
DataSetName = 'ChIP-exo'
NeuralNetworkName = 'DeepSNR'
'''
    Initialization End    
'''

FeatureMatrix, DenseLabel = Dataset.DataReader.DataReader(TFName=TFsName, DataSetName=DataSetName)
FeatureMatrix, DenseLabel = np.array(FeatureMatrix), np.array(DenseLabel)

CrossFold = sklearn.model_selection.KFold(n_splits=5)
CurrentFold = 1

ThresholdValue = 0.5

for TrainIndex, TestIndex in CrossFold.split(FeatureMatrix):
    if NeuralNetworkName == 'DeepSNR':
        NeuralNetwork = Model.DeepSNR.DeepSNR(SequenceLength=100, MotifLength=15)
    else:
        NeuralNetwork = Model.D_AEDNet.D_AEDNN(SequenceLength=100)
    TestFeatureMatrix = torch.tensor(FeatureMatrix[TestIndex], dtype=torch.float32).unsqueeze(dim=1)
    TestDenseLabels = torch.tensor(DenseLabel[TestIndex])
    TestLoader = Dataset.DataLoader.SampleLoader(FeatureMatrix=TestFeatureMatrix, DenseLabel=TestDenseLabels, BatchSize=8)

    Performance = np.zeros(shape=(5, len(TestLoader)), dtype=np.float32)

    NeuralNetwork.load_state_dict(torch.load('Weight/' + NeuralNetworkName + DataSetName + TFsName + '%dFold.pth' % CurrentFold))
    NeuralNetwork.eval()
    for step, data in enumerate(TestLoader, start=0):
        X, Y = data
        Logits = NeuralNetwork(X)
        Prediction = Utils.Threshold.Threshold(YPredicted=Logits, ThresholdValue=ThresholdValue)
        Performance[0][step], Performance[1][step], Performance[2][step], Performance[3][step], Performance[4][step] \
        = Utils.Metrics.EvaluationMetrics(y_pred=Prediction, y_true=Y.numpy())
    Performance = Performance.mean(axis=1)
    print('Acc=%.3f, Pre=%.3f, Rec=%.3f, F1-S=%.3f, Spe=%.3f' % (Performance[0], Performance[1], Performance[2], Performance[3], Performance[4]))
    CurrentFold = CurrentFold + 1