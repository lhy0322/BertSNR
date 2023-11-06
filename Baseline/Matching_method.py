import math
import os
import re
import numpy as np
import Utils.Metrics

# This function encodes a DNA nucleotide as an integer between 0 and 3
# It returns the encoded value, or -1 if the input is not a valid nucleotide.
# It is useful for later and building matrixes.
def encode(nucleotide): 
    """
    Arguments:
        nucleotide: A string of one character
    Returns:
        An integer encoding the nucleotide, or -1 if not a valid nucleotide
    """
    nucleic_acids = ["A", "C", "G", "T" ]
    if nucleotide in nucleic_acids:
        return nucleic_acids.index(nucleotide)  #if the input is a nucleic acid, it's now represented numerically by its index
    else:
        return -1

# This function builds and returns a Position Frequency Matrix from the list of sequences provided as input.
# The PFM is stored in the form of a list of lists.

def build_PFM(sequences):
    """
    Arguments:
        sequences: A list of sequences of equal lengths
    Returns:
        The position Frequency Matrix build from the sequences, stored
        as a two-dimensional list
    """
    PFM = [[0 for a in range(len(sequences[0]))] for b in range(4)]  #builds matrix
    for i in range(len(sequences)):                 
        for j in range(len(PFM[0])):                
            PFM[encode(sequences[i][j])][j] +=1     
    return PFM                                      

# This function builds and returns a PWM from a PFM and a pseudocount value. 
# The PWM is stored as a list of lists.
def get_PWM_from_PFM(PFM, pseudocount):
    """
    Arguments:
        PFM: A position frequency matrix, stored as a two-dimensional list
        pseudocount: A non-negative floating point number
    Returns:
        A position weight matrix, stored as a two-dimensional list
    """
    PWM = [[0 for a in range(len(PFM[0]))] for b in range(4)]
    for i in range(len(PFM)):         # looping through the list of list
        for j in range(len(PFM[0])):  # looping through list of values of each row
            PWM[i][j] = math.log10((PFM[i][j] + pseudocount)/((PFM[0][j]+PFM[1][j]+PFM[2][j]+PFM[3][j])+ (4 * pseudocount)))- math.log10(0.25)
    return PWM  #returns the PWM list

# This function calculates and returns the score, or likelihood a TF will bind, of a given sequence with given a PWM
def score(sequence, PWM):
    """
    Arguments:
        sequence: A DNA sequence
        PWM: A position weight matrix, of the same length as the sequence
    Returns:
        A floating point number corresponding to the score of the sequence 
        for the given PWM
    """
    score = 0                      #initial score set to zero
    for i in range(len(sequence)): #iterates over the length of the sequence which is = to the # of columns in PWM
        score = score+PWM[encode(sequence[i])][i] #uses encoded nucleotides to retrieve the score from their row for each position in the sequence
    return score                                  
        
# This function identifies and returns the list of positions in the given sequence where the PWM score is larger or equal to the threshold
def predict_sites(sequence, PWM, threshold = 0):
    """
    Arguments:
        sequence: A DNA sequence
        PWM: A position weight matrix
        threshold (optional): Minimum score needed to be predicted as a binding site
    Returns:
        A list of positions with match scores greater or equal to threshold
    """
    hits = []                            #list of hits to store calculated values
    L = len(PWM[0])                      #code will score subunits of desired length
    for i in range(len(sequence) - L+1): #iterates over the length of the sequence
        if score(sequence[i:i+L],PWM) >= threshold: #calculates hits
            hits.append(i)                          #adds hits to list
    return hits
def load_PFM(PFM_path):

    PFM = open(PFM_path, 'r').readlines()

    TF_name = PFM[0].strip('\n').split('\t')[1].upper()
    A = re.sub(' +', ' ', PFM[1]).strip('\n').split(' ')[2:-1]
    T = re.sub(' +', ' ', PFM[2]).strip('\n').split(' ')[2:-1]
    C = re.sub(' +', ' ', PFM[3]).strip('\n').split(' ')[2:-1]
    G = re.sub(' +', ' ', PFM[4]).strip('\n').split(' ')[2:-1]

    PFM_matrix = A
    if len(A) == len(T) and len(C) == len(G):
        PFM_matrix = A + T + C + G
        PFM_matrix = np.array(PFM_matrix)
        PFM_matrix = PFM_matrix.astype(int)
        PFM_matrix = PFM_matrix.reshape((4, -1))
    else:
        print(PFM_path, TF_name)

    return TF_name, PFM_matrix


if __name__ == "__main__":

    PFM_path = '../Dataset/JASPAR/PFM'
    TF_path = '../Dataset/ChIP-seq'

    result = open('Result/Result_Match.txt', 'a')
    result.write('TF_name\tAcc\tPre\tRec\tF1-S\n')

    threshold = 3

    for file_tf in os.listdir(TF_path):

        for file_pfm in os.listdir(PFM_path):

            PFM_file = PFM_path + '/' + file_pfm
            TF_name, PFM_matrix = load_PFM(PFM_file)
            TF_name = re.sub('[-:/]', '', TF_name)

            if TF_name == file_tf:
                print(TF_name)
                sequence_list = open(TF_path + '/' + file_tf + '/baseline/seq_test.txt', 'r').readlines()
                denselabel_list = open(TF_path + '/' + file_tf + '/baseline/lab_test.txt', 'r').readlines()

                PWM_matrix = get_PWM_from_PFM(PFM_matrix, 0.1)

                pred_list = np.array([])
                label_list = np.array([])

                for i in range(len(sequence_list)):
                    site_start_list = predict_sites(sequence_list[i].rstrip('\n'), PWM_matrix, threshold)

                    pred = [0] * 100
                    site_len = len(PWM_matrix[0])
                    for site_start in site_start_list:
                        site_end = site_start + site_len
                        pred[site_start:site_end] = [1] * site_len
                    pred_list = np.append(pred_list, pred)
                    label_list = np.append(label_list, list(map(int, [label for label in denselabel_list[i].rstrip('\n')])))

                Performance = np.zeros(shape=4, dtype=np.float32)

                Performance[0], Performance[1], Performance[2], Performance[3] \
                    = Utils.Metrics.EvaluationMetricsMatch(y_pred=pred_list, y_true=label_list)

                print('Acc=%.3f, Pre=%.3f, Rec=%.3f, F1-S=%.3f,' % (
                    Performance[0], Performance[1], Performance[2], Performance[3]))
                result.write(TF_name + '\t' + str(Performance[0]) + '\t' + str(Performance[1]) + '\t'
                             + str(Performance[2]) + '\t' + str(Performance[3]) + '\n')
                break

    # example code!
    
    # sites = ["ACGATG","ACAATG","ACGATC","ACGATC","TCGATC",
    #          "TCGAGC","TAGATC","TAAATC","AAAATC","ACGATA"]
    # sequence = "GCATCGATGGCAGCGACTACAGCGCTACTACAGCGGAGACGATGCGATCGATACAAT"
    #
    # print("**** Q1 ****")
    # n = encode("G")
    # print(n)
    # n = encode("Z")
    # print(n)
    #
    # print("**** Q2 ****")
    # PFM = build_PFM(sites)
    # print(PFM)
    #
    # print("**** Q3 ****")
    # my_PFM = [[6, 3, 3, 10, 0, 1],
    #           [0, 7, 0, 0, 0, 7],
    #           [0, 0, 7, 0, 1, 2],
    #           [4, 0, 0, 0, 9, 0]]
    #
    # PWM = get_PWM_from_PFM(PFM_matrix,0.1)
    # print(PWM)
    #
    # print("**** Q4 ****")
    # my_PWM=[[0.370356487039949, 0.07638834586345467, 0.07638834586345467, 0.5893480258118245, -1.4149733479708178, -0.37358066281259295], [-1.4149733479708178, 0.43628500074825727, -1.4149733479708178, -1.4149733479708178, -1.4149733479708178, 0.43628500074825727], [-1.4149733479708178, -1.4149733479708178, 0.43628500074825727, -1.4149733479708178, -0.37358066281259295, -0.09275405323689867], [0.19781050874891748, -1.4149733479708178, -1.4149733479708178, -1.4149733479708178, 0.5440680443502756, -1.4149733479708178]]
    #
    # s= score("TCGATG",PWM)
    # print(s)
    #
    # print("**** Q5 ****")
    # hits = predict_sites(sequence, PWM)
    # print(hits)

