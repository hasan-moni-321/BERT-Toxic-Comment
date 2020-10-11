import config
import transformers
import torch
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)


    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output

'''        

    def forward(self, ids, mask, token_type_ids):
        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)

        mean_pooling = torch.mean(o1, 1)
        max_pooling = torch.max(o1, 1)
        cat = torch.cat(mean_pooling, max_pooling)

        bo = self.bert_drop(cat)
        output = self.out(bo)
        return output

'''


