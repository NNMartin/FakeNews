import torch
import torch.nn as nn
from transformers import BertModel

args = {
    'learn_rate': 1e-5,
    'batch_size': 1,
    'val_batch_size': 16,
    'epochs': 5,
    'seed': 10,
    'max_len': 200,
    'classes': 2,
    'bert_type': 'bert-base-cased',
    'tokenizer': BertTokenizer.from_pretrained('bert-base-cased'),
    'num_workers': 0
}

class SentimentClassifier(nn.Module):
    def __init__(self, args):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(args['bert_type'])
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, args['classes'])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _,pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.dropout(pooled_output)
        output = self.out(output)
        return self.softmax(output)
