from transformers import BertTokenizer, BertModel
from bertviz import model_view

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

input_text = "CTAGCTGATAACTAAGGGGTTAATGAATAATCACAGTGATGAGCTCTGGAGGAGCCATGGCCCAGAGCAGCTGTGGCCTCCTTTGATTAAACCACAGAAG"

model_path = "../Main/ModelWeight/multiModel/ASCL1"
model = BertModel.from_pretrained(model_path, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)

raw_sentence = input_text
sentence_a = seq2kmer(raw_sentence, 3)

inputs = tokenizer.encode(sentence_a, return_tensors='pt')  # Tokenize input text
outputs = model(inputs)  # Run model
attention = outputs[-1]  # Retrieve attention from model outputs
tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
model_view(attention, tokens)  # Display model view