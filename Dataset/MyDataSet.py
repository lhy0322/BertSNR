from torch.utils.data import dataset


class MyDataSet(dataset.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)

class MyDataSetPredict(dataset.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class MyDataSetBERT(dataset.Dataset):
    def __init__(self, sequence, denseLabel, label):
        self.sequence = sequence
        self.denseLabel = denseLabel
        self.label = label

    def __getitem__(self, item):
        return self.sequence[item], self.denseLabel[item], self.label[item]

    def __len__(self):
        return len(self.denseLabel)


class MyDataSetSequence(dataset.Dataset):
    def __init__(self, sequence, label):
        self.sequence = sequence
        self.label = label

    def __getitem__(self, item):
        return self.sequence[item], self.label[item]

    def __len__(self):
        return len(self.label)


class MyDataSetPredictBERT(dataset.Dataset):
    def __init__(self, sequence):
        self.sequence = sequence

    def __getitem__(self, item):
        return self.sequence[item]

    def __len__(self):
        return len(self.sequence)