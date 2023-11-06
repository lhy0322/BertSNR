import torch
import numpy as np


def Threshold(YPredicted, ThresholdValue):
    ones = torch.ones_like(YPredicted)
    zeros = torch.zeros_like(YPredicted)
    output = torch.where(torch.gt(YPredicted, ThresholdValue), ones, zeros)
    return output.numpy().astype(np.int32)
