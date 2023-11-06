from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import transformers.optimization as op
import pandas as pd
import os

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
DataSetName = 'ChIP-seq'
set_seed(1)
use_gpu = torch.cuda.is_available()
print(use_gpu)

KMER = 3
MaxEpoch = 10
BatchSize = 32
t = 0.15
ThresholdValue = 0.5
'''
    Initialization End    
'''

'''
    Main Process as follows: 
'''
result_sequence = open('Result/Result_TFBert_sequence.txt', 'a')
result_sequence.write('TF_name\tAcc\tPre\tRec\tF1-S\tAUC\tAUPR\n')
result_token = open('Result/Result_TFBert_token.txt', 'a')
result_token.write('TF_name\tAcc\tPre\tRec\tF1-S\tAUC\tAUPR\n')

for TF in os.listdir('../Dataset/ChIP-seq'):
    TFsName = TF

    print(TFsName)
    acc_token = 0
    pre_token = 0
    rec_token = 0
    f1_token = 0
    auc_token = 0
    aupr_token = 0

    acc_sequence = 0
    pre_sequence = 0
    rec_sequence = 0
    f1_sequence = 0
    auc_sequence = 0
    aupr_sequence = 0

    # os.makedirs('ModelWeight/multiModel/' + TFsName)
    TrainSequence, TrainDenseLabel, TrainLabel, TestSequence, TestDenseLabel, TestLabel = \
        Dataset.DataReader.DataReaderBERT(TFsName, DataSetName, KMER)
    TrainSequence, TrainDenseLabel, TrainLabel, TestSequence, TestDenseLabel, TestLabel = \
        np.array(TrainSequence), np.array(TrainDenseLabel), np.array(TrainLabel), \
        np.array(TestSequence), np.array(TestDenseLabel), np.array(TestLabel)

    LossFunction = nn.BCELoss().cuda()

    NeuralNetwork = Model.BertSNR.BERTSNR(KMER).cuda()
    t_total = len(TrainSequence) * MaxEpoch // BatchSize

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in NeuralNetwork.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in NeuralNetwork.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=2e-4, eps=1e-8,
                            betas=(0.9, 0.999))
    scheduler = op.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total
    )

    TrainDenseLabels = torch.tensor(TrainDenseLabel)
    TrainLabels = torch.tensor(TrainLabel)
    TrainLoader = Dataset.DataLoader.SampleLoaderBERT(Sequence=TrainSequence, DenseLabel=TrainDenseLabels,
                                                      Label=TrainLabels, BatchSize=BatchSize)

    TestDenseLabels = torch.tensor(TestDenseLabel)
    TestLabels = torch.tensor(TestLabel)
    TestLoader = Dataset.DataLoader.SampleLoaderBERT(Sequence=TestSequence, DenseLabel=TestDenseLabel,
                                                     Label=TestLabel, BatchSize=BatchSize)

    earlystop = 0
    for Epoch in range(MaxEpoch):
        # train
        NeuralNetwork.train()
        TrainProgressBar = tqdm(TrainLoader)
        for data in TrainProgressBar:
            optimizer.zero_grad()
            X, Y1, Y2 = data
            X = X
            Y1 = Y1.cuda()
            Y2 = Y2.cuda()
            Prediction_sequence, Prediction_token = NeuralNetwork(X)
            Loss_sequence = LossFunction(Prediction_sequence.squeeze(), Y2.to(torch.float32))
            Loss_token = LossFunction(Prediction_token.squeeze(), Y1.to(torch.float32))
            # Linear interpolation
            Loss = t * Loss_sequence + (1 - t) * Loss_token
            # Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
            # Loss = torch.exp(-log_var_a) * Loss_sequence + torch.exp(-log_var_b) * Loss_token + log_var_a + log_var_b
            TrainProgressBar.set_description(f'Epoch {Epoch} loss:{Loss.cpu().item()}')
            Loss.backward()
            optimizer.step()
            scheduler.step()
        # valid
        NeuralNetwork.eval()
        ValidProgressBar = tqdm(TestLoader)

        pred_sequence = np.array([])
        label_sequence = np.array([])
        logits_sequence = np.array([])

        pred_token = np.array([])
        label_token = np.array([])
        logits_token = np.array([])

        for data in ValidProgressBar:
            X, Y1, Y2 = data
            X = X
            Y1 = Y1.cuda()
            Y2 = Y2.cuda()
            Logits_sequence, Logits_token = NeuralNetwork(X)

            Prediction_sequence = Utils.Threshold.Threshold(YPredicted=Logits_sequence.cpu(),
                                                            ThresholdValue=ThresholdValue)
            Prediction_token = Utils.Threshold.Threshold(YPredicted=Logits_token.cpu(),
                                                         ThresholdValue=ThresholdValue)

            logits_sequence = np.append(logits_sequence, Logits_sequence.cpu().detach().numpy())
            pred_sequence = np.append(pred_sequence, Prediction_sequence)
            label_sequence = np.append(label_sequence, Y2.cpu())

            logits_token = np.append(logits_token, Logits_token.cpu().detach().numpy())
            pred_token = np.append(pred_token, Prediction_token)
            label_token = np.append(label_token, Y1.cpu())

        Performance_sequence = np.zeros(shape=6, dtype=np.float32)
        Performance_token = np.zeros(shape=6, dtype=np.float32)

        # sequence
        Performance_sequence[0], Performance_sequence[1], Performance_sequence[2], \
        Performance_sequence[3], Performance_sequence[4], Performance_sequence[5] \
            = Utils.Metrics.EvaluationMetricsSequence(y_pred=pred_sequence, y_true=label_sequence,
                                                      y_logits=logits_sequence)
        # token
        Performance_token[0], Performance_token[1], Performance_token[2], \
        Performance_token[3], Performance_token[4], Performance_token[5] \
            = Utils.Metrics.EvaluationMetricsToken(y_pred=pred_token, y_true=label_token, y_logits=logits_token,
                                                   kmer=KMER)
        # print(Performance_sequence)
        # print(Performance_token)
        if Performance_token[5] > aupr_token:
            acc_sequence = Performance_sequence[0]
            pre_sequence = Performance_sequence[1]
            rec_sequence = Performance_sequence[2]
            f1_sequence = Performance_sequence[3]
            auc_sequence = Performance_sequence[4]
            aupr_sequence = Performance_sequence[5]

            acc_token = Performance_token[0]
            pre_token = Performance_token[1]
            rec_token = Performance_token[2]
            f1_token = Performance_token[3]
            auc_token = Performance_token[4]
            aupr_token = Performance_token[5]

            # torch.save(NeuralNetwork.state_dict(), 'ModelWeight/multiModel/' + TFsName + '/pytorch_model.bin')
        else:
            earlystop += 1
            if earlystop > 2:
                break

    print('Sequence Classification: Acc=%.3f, Pre=%.3f, Rec=%.3f, F1-S=%.3f, AUC=%.3f, AUPR=%.3f,' % (
        acc_sequence, pre_sequence, rec_sequence, f1_sequence, auc_sequence, aupr_sequence))
    print('Token Classification: Acc=%.3f, Pre=%.3f, Rec=%.3f, F1-S=%.3f, AUC=%.3f, AUPR=%.3f,' % (
        acc_token, pre_token, rec_token, f1_token, auc_token, aupr_token))
    result_sequence.write(TFsName + '\t' + str(acc_sequence) + '\t' + str(pre_sequence) + '\t' + str(rec_sequence) +
                          '\t' + str(f1_sequence) + '\t' + str(auc_sequence) + '\t' + str(aupr_sequence) + '\n')
    result_token.write(TFsName + '\t' + str(acc_token) + '\t' + str(pre_token) + '\t' + str(rec_token) +
                          '\t' + str(f1_token) + '\t' + str(auc_token) + '\t' + str(aupr_token) + '\n')
    result_sequence.flush()
    result_token.flush()



