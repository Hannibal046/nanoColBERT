import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertPreTrainedModel,
    BertModel,
    BertTokenizerFast,
)
import string

class ColBERT(BertPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)

        self.bert = BertModel(config,add_pooling_layer=False)
        self.linear = nn.Linear(config.hidden_size,config.dim,bias=False)
        
        self.similarity_metric= config.similarity_metric
        self.mask_punctuation = config.mask_punctuation

        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        mask_symbol_list = [tokenizer.pad_token_id]
        if self.mask_punctuation:    
            mask_symbol_list += [tokenizer.encode(symbol,add_special_tokens=False)[0] for symbol in string.punctuation]
        self.register_buffer('mask_buffer', torch.tensor(mask_symbol_list)) 

        self.init_weights()
    
    def get_query_embedding(self,input_ids,attention_mask):
        query_embedding = self.bert(
            input_ids,attention_mask,
        ).last_hidden_state
        query_embedding = self.linear(query_embedding)
        query_embedding = F.normalize(query_embedding,p=2,dim=2)
        return query_embedding
    
    def get_doc_embedding(self,input_ids,attention_mask,return_list=False):
        doc_embedding = self.bert(
            input_ids,attention_mask,
        ).last_hidden_state
        doc_embedding = self.linear(doc_embedding)
        
        
        puntuation_mask = self.punctuation_mask(input_ids).unsqueeze(2)
        doc_embedding = doc_embedding * puntuation_mask

        doc_embedding = F.normalize(doc_embedding,p=2,dim=2)
        
        if not return_list:
            return doc_embedding
        else:
            doc_embedding = doc_embedding.cpu().to(dtype=torch.float16)
            puntuation_mask = puntuation_mask.cpu().bool().squeeze(-1)
            doc_embedding = [d[puntuation_mask[idx]] for idx,d in enumerate(doc_embedding)]
            return doc_embedding
        
    
    def punctuation_mask(self,input_ids):
        mask = (input_ids.unsqueeze(-1) == self.mask_buffer).any(dim=-1)
        mask = (~mask).float()
        return mask

    def forward(
        self,
        query_input_ids, # [bs,seq_len]
        query_attention_mask, # [bs,seq_len]

        doc_input_ids, # [bs*2,seq_len]
        doc_attention_mask, # [bs*2,seq_len]
    ):  
        query_embedding = self.get_query_embedding(query_input_ids,query_attention_mask)
        query_embedding = query_embedding.repeat(2,1,1)
        doc_embedding   = self.get_doc_embedding(doc_input_ids,doc_attention_mask)

        return self.score(query_embedding,doc_embedding)
    
    def score(self,query_embedding,doc_embedding):
        if self.similarity_metric == 'cosine':
            return (query_embedding @ doc_embedding.permute(0, 2, 1)).max(2).values.sum(1)

        elif self.similarity_metric == 'l2':
            return (-1.0 * ((query_embedding.unsqueeze(2) - doc_embedding.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)