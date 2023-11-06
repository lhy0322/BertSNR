import scipy.io
import os
import pandas as pd
from Utils.Shuffle import dinuclShuffle


def seq2kmer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers


def deleteByStart(s, start, kmer):
    # 找出字符串在原始字符串中的位置，开始位置是：开始始字符串的最左边第一个位置
    x1 = s.index(start)
    # 找出两个字符串的内容
    x2 = s[:x1]
    x3 = s[x1+kmer-1:]
    # 将内容替换为控制符串
    result = x2 + x3
    return result


fileList = os.listdir('ChIP-seq')

for file in fileList:
    print(file)
    dataset_path = 'ChIP-seq/' + file

    sequence = pd.read_csv(dataset_path + '/sequence.txt', header=None)
    denseLabel = pd.read_csv(dataset_path + '/label.txt', header=None)

    baseline_path = dataset_path + '/baseline'
    os.makedirs(baseline_path)

    seq_train = open(baseline_path + '/seq_train.txt', 'a')
    seq_test = open(baseline_path + '/seq_test.txt', 'a')
    lab_train = open(baseline_path + '/lab_train.txt', 'a')
    lab_test = open(baseline_path + '/lab_test.txt', 'a')

    for i in range(len(sequence)):

        if i < len(sequence) // 5:
            seq_test.write(str(sequence[0][i]) + '\n')
            lab_test.write(str(denseLabel[0][i]) + '\n')

        else:
            seq_train.write(str(sequence[0][i]) + '\n')
            lab_train.write(str(denseLabel[0][i]) + '\n')

    seq_train.close()
    seq_test.close()
    lab_train.close()
    lab_test.close()

    for kmer in range(3, 7):
        out_path = dataset_path + '/' + str(kmer) + '-mer'
        os.makedirs(out_path)

        train = open(out_path + '/train.txt', 'a')
        train.write('sequence' + '\t' + 'denseLabel' + '\t' + 'label' + '\n')
        test = open(out_path + '/test.txt', 'a')
        test.write('sequence' + '\t' + 'denseLabel' + '\t' + 'label' + '\n')

        for i in range(len(sequence)):

            seq = str(sequence[0][i])
            kmer_seq = seq2kmer(seq, kmer)
            if i % 2 == 0:
                dense = deleteByStart(str(denseLabel[0][i]), '1', kmer)
                label = '1'
            else:
                dense = str(denseLabel[0][i])[kmer-1:]
                label = '0'

            if i < len(sequence) // 5:
                test.write(kmer_seq + '\t' + dense + '\t' + label + '\n')
            else:
                train.write(kmer_seq + '\t' + dense + '\t' + label + '\n')
        train.close()
        test.close()






