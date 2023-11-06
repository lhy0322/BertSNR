from Dataset import MyDataSet
from torch.utils.data import DataLoader


def SampleLoader(FeatureMatrix, DenseLabel, BatchSize):
    Loader = DataLoader(
        dataset=MyDataSet.MyDataSet(FeatureMatrix, DenseLabel),
        batch_size=BatchSize,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )
    return Loader

def SampleLoaderPredict(FeatureMatrix, BatchSize):
    Loader = DataLoader(
        dataset=MyDataSet.MyDataSetPredict(FeatureMatrix),
        batch_size=BatchSize,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )
    return Loader

def SampleLoaderBERT(Sequence, DenseLabel, Label, BatchSize):
    Loader = DataLoader(
        dataset=MyDataSet.MyDataSetBERT(Sequence, DenseLabel, Label),
        batch_size=BatchSize,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )
    return Loader

def SampleLoaderSequence(Sequence, Label, BatchSize):
    Loader = DataLoader(
        dataset=MyDataSet.MyDataSetSequence(Sequence, Label),
        batch_size=BatchSize,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )
    return Loader

def SampleLoaderPredictBERT(Sequence, DenseLabel, Label, BatchSize):
    Loader = DataLoader(
        dataset=MyDataSet.MyDataSetBERT(Sequence, DenseLabel, Label),
        batch_size=BatchSize,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    return Loader

def SampleLoaderPredictUnlabelBERT(Sequence, BatchSize):
    Loader = DataLoader(
        dataset=MyDataSet.MyDataSetPredictBERT(Sequence),
        batch_size=BatchSize,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    return Loader