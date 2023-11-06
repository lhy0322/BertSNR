import torch.nn as nn
import torch


class AttentionGates(nn.Module):
    def __init__(self, DimOfGS, DimOfSC, DimOfAG):
        super(AttentionGates, self).__init__()
        self.ParallelConvolutionOfGS = nn.Sequential(
            nn.Conv2d(DimOfGS, DimOfAG, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(DimOfAG)
        )
        self.ParallelConvolutionOfSC = nn.Sequential(
            nn.Conv2d(DimOfGS, DimOfAG, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(DimOfAG)
        )
        self.Relu = nn.ReLU()
        self.ComputeWeightMap = nn.Sequential(
            nn.Conv2d(DimOfAG, 1, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, GatingSignal, SkipConnection):
        NewGS = self.ParallelConvolutionOfGS(GatingSignal)
        NewSC = self.ParallelConvolutionOfSC(SkipConnection)
        Integration = self.Relu(NewGS + NewSC)
        WeightMap = self.ComputeWeightMap(Integration)
        Output = SkipConnection * WeightMap
        return Output


class D_AEDNN(nn.Module):
    def __init__(self, SequenceLength):
        super(D_AEDNN, self).__init__()
        self.StackConvolutionOfEB1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.DownSampleOfEB1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.StackConvolutionOfEB2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.DownSampleOfEB2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.BridgeNetwork = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.UpSampleOfDB2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), stride=(1, 1)),
            nn.UpsamplingBilinear2d(size=(2, int(0.5 * SequenceLength)))
        )
        self.AttentionGate2 = AttentionGates(DimOfGS=32, DimOfSC=32, DimOfAG=16)
        self.StackConvolutionOfDB2 = nn.Sequential(
            nn.Conv2d(in_channels=32 + 32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.UpSampleOfDB1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), stride=(1, 1)),
            nn.UpsamplingBilinear2d(size=(4, SequenceLength))
        )
        self.AttentionGate1 = AttentionGates(DimOfGS=16, DimOfSC=32, DimOfAG=16)
        self.StackConvolutionOfDB1 = nn.Sequential(
            nn.Conv2d(in_channels=16 + 16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.OutputNetwork = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(4, 1), padding=0),
            nn.Sigmoid()
        )

    def forward(self, ModelInput):
        EncoderNetworkConv1 = self.StackConvolutionOfEB1(ModelInput)
        EncoderNetworkDown1 = self.DownSampleOfEB1(EncoderNetworkConv1)
        EncoderNetworkConv2 = self.StackConvolutionOfEB2(EncoderNetworkDown1)
        EncoderNetworkDown2 = self.DownSampleOfEB2(EncoderNetworkConv2)
        BridgeNetworkOut = self.BridgeNetwork(EncoderNetworkDown2)
        DecoderNetworkUp2 = self.UpSampleOfDB2(BridgeNetworkOut)
        GateOut2 = self.AttentionGate2(DecoderNetworkUp2, EncoderNetworkConv2)
        DecoderNetworkConv2 = self.StackConvolutionOfDB2(torch.cat((GateOut2, DecoderNetworkUp2), dim=1))
        DecoderNetworkUp1 = self.UpSampleOfDB1(DecoderNetworkConv2)
        GateOut1 = self.AttentionGate1(DecoderNetworkUp1, EncoderNetworkConv1)
        DecoderNetworkConv1 = self.StackConvolutionOfDB1(torch.cat((GateOut1, DecoderNetworkUp1), dim=1))
        ModelOutput = self.OutputNetwork(DecoderNetworkConv1)
        return ModelOutput
