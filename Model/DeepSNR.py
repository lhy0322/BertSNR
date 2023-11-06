import torch.nn as nn
import torch


class ConvolutionalNetwork(nn.Module):
    def __init__(self, MotifLength):
        super(ConvolutionalNetwork, self).__init__()
        self.Convolution = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, MotifLength), padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.GlobalMaxPool = nn.AdaptiveMaxPool2d(output_size=(1, 1), return_indices=True)
        self.FullyConnection = nn.Sequential(
            nn.Linear(16, 1),
            nn.ReLU(),
            nn.Linear(1, 16)
        )

    def forward(self, x):
        output = self.Convolution(x)
        output, indices = self.GlobalMaxPool(output)
        output = torch.flatten(output, start_dim=1)
        output = self.FullyConnection(output)
        return output, indices


class DeconvolutionalNetwork(nn.Module):
    def __init__(self, SequenceLength, MotifLength):
        super(DeconvolutionalNetwork, self).__init__()
        self.GlobalMaxUnPool = nn.MaxUnpool2d(kernel_size=(1, SequenceLength - MotifLength + 1), stride=(1, 1))
        self.Deconvolution = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(4, MotifLength), stride=(1, 1), padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

    def forward(self, x, indices):
        x = x.view([-1, 16, 1, 1])
        output = self.GlobalMaxUnPool(x, indices)
        output = self.Deconvolution(output)
        return output


class OutputNetwork(nn.Module):
    def __init__(self):
        super(OutputNetwork, self).__init__()
        self.OutPut = nn.Sequential(
            nn.MaxPool2d(kernel_size=(4, 1), stride=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.OutPut(x)
        return output


class DeepSNR(nn.Module):
    def __init__(self, SequenceLength, MotifLength):
        super(DeepSNR, self).__init__()
        self.ConvolutionalNetwork = ConvolutionalNetwork(MotifLength=MotifLength)
        self.DeconvolutionalNetwork = DeconvolutionalNetwork(SequenceLength=SequenceLength, MotifLength=MotifLength)
        self.OutputNetwork = OutputNetwork()

    def forward(self, x):
        ConvOut, indices = self.ConvolutionalNetwork(x)
        DeconvOut = self.DeconvolutionalNetwork(ConvOut, indices)
        Prediction = self.OutputNetwork(DeconvOut)
        return Prediction