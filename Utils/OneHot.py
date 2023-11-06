import numpy as np

'''
OneHot将对文件中所有数据进行编码
输入参数:
    sequence表示全部核苷酸序列数据
    number表示样本的个数
    nucleotide表示碱基的种类，DNA中通常为 A T C G 四种
    length表示某个样本的碱基个数
'''


def OneHot(sequence, number, nucleotide, length):
    code = np.zeros(shape=(number, nucleotide, length), dtype=np.float32)
    for i in range(number):
        line = sequence[i]
        for j in range(length):
            if line[j] == 'A':
                code[i][0][j] = 1.0
                code[i][1][j] = 0.0
                code[i][2][j] = 0.0
                code[i][3][j] = 0.0
            elif line[j] == 'C':
                code[i][0][j] = 0.0
                code[i][1][j] = 1.0
                code[i][2][j] = 0.0
                code[i][3][j] = 0.0
            elif line[j] == 'G':
                code[i][0][j] = 0.0
                code[i][1][j] = 0.0
                code[i][2][j] = 1.0
                code[i][3][j] = 0.0
            else:
                code[i][0][j] = 0.0
                code[i][1][j] = 0.0
                code[i][2][j] = 0.0
                code[i][3][j] = 1.0
    return code
