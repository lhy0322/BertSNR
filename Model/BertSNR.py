from transformers import BertModel, BertTokenizer
import torch.nn as nn


class BERTSNR(nn.Module):
    def __init__(self, kmer):
        super(BERTSNR, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('../DNABERT/'+str(kmer)+'-new-12w-0', do_lower_case=False)
        self.bert = BertModel.from_pretrained('../DNABERT/'+str(kmer)+'-new-12w-0')
        self.dropout = nn.Dropout(0.2)
        # self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size,
        #                     bidirectional=True,
        #                     num_layers=1,
        #                     hidden_size=768//2,
        #                     batch_first=True)
        self.classifier_token = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )
        self.classifier_sequence = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

    def forward(self, input_seq):

        encoded_input = self.tokenizer(input_seq, return_tensors='pt')

        for key in encoded_input:
            encoded_input[key] = encoded_input[key].cuda()
        output = self.bert(**encoded_input)

        # sequence classification
        cls = output[1]
        cls = self.dropout(cls)
        logits_sequence = self.classifier_sequence(cls)

        # token classification
        token = output[0]
        token = token[:, 1:-1, :]
        # token, _ = self.lstm(token)
        token = self.dropout(token)
        logits_token = self.classifier_token(token)

        return logits_sequence, logits_token
