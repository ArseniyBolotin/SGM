import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class BertEncdoer(nn.Module):
    def __init__(self, hidden_size, bert_type="bert-base-uncased", freeze_bert_weights=False):
        super(BertEncdoer, self).__init__()
        config = BertConfig.from_pretrained(bert_type, output_hidden_states=True)
        config.return_dict = True
        self.bert_model = BertModel.from_pretrained(bert_type, config=config)
        self.hidden_size = hidden_size
        if freeze_bert_weights: # == false !DocBERT
            for param in self.bert_model.parameters():
                param.requires_grad = False
        self.n_bert_features = self.bert_model.pooler.dense.in_features
        self.dense = nn.Linear(self.n_bert_features, self.hidden_size)

    def forward(self, inputs, lens):
        bert_output = self.bert_model(**inputs)
        encoder_output = self.dense(bert_output.last_hidden_state)
        return encoder_output.transpose(0, 1), (torch.zeros(1, encoder_output.size(0), self.hidden_size, device='cuda'), torch.zeros(1, encoder_output.size(0), self.hidden_size, device='cuda'))
