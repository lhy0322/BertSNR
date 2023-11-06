import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np
from transformers import BertTokenizer, BertModel


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


def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)


def get_attention_dna(model, tokenizer, sentence_a, start, end):
    inputs = tokenizer.encode_plus(sentence_a, sentence_b=None, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids']
    attention = model(input_ids)[-1]
    input_id_list = input_ids[0].tolist()  # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    attn = format_attention(attention)
    attn_score = []
    for i in range(1, len(tokens) - 1):
        attn_score.append(float(attn[start:end + 1, :, 0, i].sum()))
    return attn_score


def get_real_score(attention_scores, kmer, metric):
    counts = np.zeros([len(attention_scores) + kmer - 1])
    real_scores = np.zeros([len(attention_scores) + kmer - 1])

    if metric == "mean":
        for i, score in enumerate(attention_scores):
            for j in range(kmer):
                counts[i + j] += 1.0
                real_scores[i + j] += score

        real_scores = real_scores / counts
    else:
        pass

    return real_scores


SEQUENCE = "ACAACTTCTCAGTTAGACTGCGCCCCCGCTGGCAGTGAAAGGGAAGTGCAGCTGGCCAGCACTTAGAACTCCCATTACACCATCACAGCATCAAATCCGC"

# SEQUENCE = "GGAGCCATGGCCCAGAGCAGCTGTG"

def Visualize_sequences(args):

    # load model and calculate attention
    model_path = args.model_path
    model = BertModel.from_pretrained(model_path, output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)

    data = open('Visual/CDX2.txt', 'r').readlines()

    score_list = np.array([])

    for raw_sentence in data:
        raw_sentence = raw_sentence.strip('\n')
        sentence_a = seq2kmer(raw_sentence, args.kmer)

        attention = get_attention_dna(model, tokenizer, sentence_a, start=args.start_layer, end=args.end_layer)
        # attention[91] = attention[90]
        attention_scores = np.array(attention).reshape(np.array(attention).shape[0], 1)

        real_scores = get_real_score(attention_scores, args.kmer, args.metric)
        # print(real_scores.shape[0])
        scores = real_scores.reshape(1, real_scores.shape[0])
        score_list = np.append(score_list, scores)

    # print(score_list)
    plt.figure(figsize=(6.5, 1.7))
    scores = score_list.reshape(-1, 100)
    avg = np.mean(scores, axis=0).reshape(100)
    plt.ylim(0, 1.5)
    plt.yticks([0.5,1.0])
    # plt.xlabel('Distance from TFBS center (bp)', fontsize=10)
    # plt.ylabel('Avg attention', fontsize=10)
    plt.tick_params(labelsize=8)
    x = range(-50, 50)
    y = avg

    plt.plot(x, y)
    plt.show()
    print(avg)

    list1 = [''] * 32021
    list1[10000] = '10000'
    list1[20000] = '20000'
    list1[30000] = '30000'
    # plot
    # plt.figure(figsize=(15, 1))
    sns.set()
    ax = sns.heatmap(scores, cmap='YlGnBu', vmin=0, cbar=True, yticklabels=list1, xticklabels=False)
    plt.yticks(fontsize=7)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    # ax.set_yticks([100, 200])
    # plt.yticks([100, 200])
    # plt.savefig('Visual/seq3.png')
    plt.show()

def Visualize(args):

    # load model and calculate attention
    model_path = args.model_path
    model = BertModel.from_pretrained(model_path, output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
    raw_sentence = args.sequence if args.sequence else SEQUENCE
    sentence_a = seq2kmer(raw_sentence, args.kmer)

    attention = get_attention_dna(model, tokenizer, sentence_a, start=args.start_layer, end=args.end_layer)
    # attention[91] = attention[90]
    attention_scores = np.array(attention).reshape(np.array(attention).shape[0], 1)

    real_scores = get_real_score(attention_scores, args.kmer, args.metric)
    # print(real_scores.shape[0])
    scores = real_scores.reshape(1, real_scores.shape[0])

    print(scores)

    # plot
    plt.figure(figsize=(15, 1))
    sns.set()
    ax = sns.heatmap(scores, cmap='YlGnBu', vmin=0, cbar=False, yticklabels=False, xticklabels=False)
    # plt.savefig('Visual/seq3.png')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kmer",
        default=3,
        type=int,
        help="K-mer",
    )
    parser.add_argument(
        "--model_path",
        default="../Main/ModelWeight/multiModel/ASCL1",
        type=str,
        help="The path of the finetuned model",
    )
    parser.add_argument(
        "--start_layer",
        default=11,
        type=int,
        help="Which layer to start",
    )
    parser.add_argument(
        "--end_layer",
        default=11,
        type=int,
        help="which layer to end",
    )
    parser.add_argument(
        "--metric",
        default="mean",
        type=str,
        help="the metric used for integrate predicted kmer result to real result",
    )
    parser.add_argument(
        "--sequence",
        default=None,
        type=str,
        help="the sequence for visualize",
    )

    args = parser.parse_args()
    Visualize(args)
    # Visualize_sequences(args)


if __name__ == "__main__":
    main()