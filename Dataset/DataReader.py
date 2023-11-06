import scipy.io
import os
import Utils.OneHot
import pandas as pd

'''
    TFName represents name of TFs
    DataSetName represents name of Datasets
'''
def seq2kmer(seq):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x + 3] for x in range(len(seq) + 1 - 3)]
    kmers = " ".join(kmer)
    return kmers


def DataReader(TFName, DataSetName, type):
    FeatureMatrix = []
    DenseLabels = []
    if DataSetName == 'ChIP-exo':
        '''
            AR -> labels
            GR -> label
        '''
        DenseLabels = scipy.io.loadmat(os.path.dirname(__file__) + '/ChIP-exo/' + TFName + '/label.mat')['labels']
        with open(os.path.dirname(__file__) + '/ChIP-exo/' + TFName + '/sequence.txt', 'r') as SReader:
            for line in SReader.readlines():
                FeatureMatrix.append(list(map(str, line.rstrip('\n'))))
        FeatureMatrix = Utils.OneHot.OneHot(sequence=FeatureMatrix, number=len(FeatureMatrix), nucleotide=4,
                                            length=DenseLabels.shape[1])
    else:
        with open(os.path.dirname(__file__) + '/ChIP-seq/' + TFName + '/baseline/seq_' + type + '.txt', 'r') as SReader, open(
                  os.path.dirname(__file__) + '/ChIP-seq/' + TFName + '/baseline/lab_' + type + '.txt', 'r') as LReader:
            for SRLine, LRLine in zip(SReader.readlines(), LReader.readlines()):
                FeatureMatrix.append(SRLine.rstrip('\n'))
                DenseLabels.append(list(map(int, [label for label in LRLine.rstrip('\n')])))
        FeatureMatrix = Utils.OneHot.OneHot(sequence=FeatureMatrix, number=len(FeatureMatrix), nucleotide=4,
                                            length=len(DenseLabels[1]))
    return FeatureMatrix, DenseLabels

def DataReaderPrecit(path):

    Dataset = pd.read_csv(path, header=None)
    Sequence = Dataset[0].tolist()
    FeatureMatrix = Utils.OneHot.OneHot(sequence=Sequence, number=len(Sequence), nucleotide=4,
                                        length=100)
    return FeatureMatrix


def DataReaderBERT(TFName, DataSetName, KMER):

    trainDataset = pd.read_csv('../Dataset/' + DataSetName + '/' + TFName + '/' + str(KMER) + '-mer/train.txt', sep='\t')

    TrainSequence = trainDataset['sequence'].tolist()
    TrainDenseLabels = []
    TrainLabels = trainDataset['label'].tolist()

    for row in trainDataset['denseLabel']:
        TrainDenseLabels.append(list(map(int, [label for label in row])))

    testDataset = pd.read_csv('../Dataset/' + DataSetName + '/' + TFName + '/' + str(KMER) + '-mer/test.txt', sep='\t')

    TestSequence = testDataset['sequence'].tolist()
    TestDenseLabels = []
    TestLabels = testDataset['label'].tolist()

    for row in testDataset['denseLabel']:
        TestDenseLabels.append(list(map(int, [label for label in row])))

    return TrainSequence, TrainDenseLabels, TrainLabels, TestSequence, TestDenseLabels, TestLabels

def DataReaderSequence(TFName, DataSetName, KMER):

    trainDataset = pd.read_csv('../Dataset/' + DataSetName + '/' + TFName + '/' + str(KMER) + '-mer/train.txt', sep='\t')

    TrainSequence = trainDataset['sequence'].tolist()
    TrainLabels = trainDataset['label'].tolist()

    testDataset = pd.read_csv('../Dataset/' + DataSetName + '/' + TFName + '/' + str(KMER) + '-mer/test.txt', sep='\t')

    TestSequence = testDataset['sequence'].tolist()
    TestLabels = testDataset['label'].tolist()

    return TrainSequence, TrainLabels, TestSequence, TestLabels

def DataReaderPrecitBERT(path):

    Dataset = pd.read_csv(path, header=None)
    Sequence = Dataset[0].tolist()
    Sequence = list(map(seq2kmer, Sequence))

    return Sequence