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

'''
    Initialization as follows:
'''
TFsName = 'CDX2'
DataSetName = 'ChIP-seq'
use_gpu = torch.cuda.is_available()
print(use_gpu)
'''
    Initialization End    
'''

'''
    Main Process as follows: 
'''


Sequence, DenseLabel, Label = Dataset.DataReader.DataReaderBERT(TFsName, DataSetName)
Sequence, DenseLabel, Label = np.array(Sequence), np.array(DenseLabel), np.array(Label)

CrossFold = sklearn.model_selection.KFold(n_splits=5)
ThresholdValue = 0.5

LossFunction = nn.BCELoss().cuda()

MaxEpoch = 10
BatchSize = 64


for TrainIndex, ValidIndex in CrossFold.split(Sequence):
    NeuralNetwork = Model.BertSNR.BERT().cuda()
    no_decay = ["bias", "LayerNorm.weight"]
    t_total = len(TrainIndex)*MaxEpoch//BatchSize

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in NeuralNetwork.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in NeuralNetwork.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=2e-4, eps=1e-8,
                      betas=(0.9, 0.999))
    scheduler = op.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(t_total*0.1), num_training_steps=t_total
    )

    TrainSequence = Sequence[TrainIndex]
    TrainDenseLabels = torch.tensor(DenseLabel[TrainIndex])
    TrainLabels = torch.tensor(Label[TrainIndex])
    TrainLoader = Dataset.DataLoader.SampleLoaderBERT(Sequence=TrainSequence, DenseLabel=TrainDenseLabels,
                                                      Label=TrainLabels, BatchSize=BatchSize)
    ValidSequence = Sequence[ValidIndex]
    ValidDenseLabels = torch.tensor(DenseLabel[ValidIndex])
    ValidLabels = torch.tensor(Label[ValidIndex])
    ValidLoader = Dataset.DataLoader.SampleLoaderBERT(Sequence=ValidSequence, DenseLabel=ValidDenseLabels,
                                                      Label=ValidLabels, BatchSize=BatchSize)

    for Epoch in range(MaxEpoch):
        # train
        NeuralNetwork.train()
        TrainProgressBar = tqdm(TrainLoader)
        for data in TrainProgressBar:
            TrainProgressBar.set_description("Epoch %d" % Epoch)
            optimizer.zero_grad()
            X, Y1, Y2 = data
            X = X
            Y1 = Y1.cuda()
            Y2 = Y2.cuda()
            Prediction = NeuralNetwork(X)

            Loss = LossFunction(Prediction.squeeze(), Y1.to(torch.float32))

            Loss.backward()
            optimizer.step()
            scheduler.step()
        # valid
        NeuralNetwork.eval()
        ValidProgressBar = tqdm(ValidLoader)
        pred = np.array([])
        label = np.array([])
        logits = np.array([])
        for data in ValidProgressBar:
            X, Y1, Y2 = data
            X = X
            Y1 = Y1.cuda()
            Y2 = Y2.cuda()
            Logits = NeuralNetwork(X)
            Prediction = Utils.Threshold.Threshold(YPredicted=Logits.cpu(), ThresholdValue=ThresholdValue)
            logits = np.append(logits, Logits.cpu().detach().numpy())
            pred = np.append(pred, Prediction)
            label = np.append(label, Y1.cpu())
        Performance = np.zeros(shape=6, dtype=np.float32)
        Performance[0], Performance[1], Performance[2], Performance[3], Performance[4], Performance[5] \
            = Utils.Metrics.EvaluationMetrics(y_pred=pred, y_true=label, y_logits=logits)
        print('Acc=%.3f, Pre=%.3f, Rec=%.3f, F1-S=%.3f, AUC=%.3f, AUPR=%.3f,' % (
            Performance[0], Performance[1], Performance[2], Performance[3], Performance[4], Performance[5]))
    # save model
    # torch.save(NeuralNetwork.state_dict(), 'Weight/' + NeuralNetworkName + DataSetName + TFsName)

print('Finished Training')
