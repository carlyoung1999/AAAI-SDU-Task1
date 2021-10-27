'''
Description: 
Author: Li Siheng
Date: 2021-09-28 08:40:34
LastEditTime: 2021-10-27 07:31:32
'''
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModel
from torchcrf import CRF
from .base_model import BaseAEModel


class BertLSTMModel(BaseAEModel):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(args, tokenizer)
        
        if isinstance(args, dict):
            args = argparse.Namespace(**args)
            
        self.tokenizer = tokenizer
        self.nlabels = args.nlabels
        self.use_crf = args.use_crf

        self.config = AutoConfig.from_pretrained(args.pretrain_model)
        self.bert = AutoModel.from_pretrained(args.pretrain_model)
        self.bert.resize_token_embeddings(new_num_tokens=len(tokenizer))

        hidden_size = self.config.hidden_size
        self.lstm = nn.LSTM(hidden_size,
                            args.rnn_size,
                            args.rnn_nlayer,
                            bidirectional=True,
                            batch_first=True)

        self.dropout = nn.Dropout(args.ffn_dropout)
        self.ffn = nn.Linear(2 * args.rnn_size, self.nlabels, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

        if self.use_crf:
            self.crf = CRF(self.nlabels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids, attention_mask, token_type_ids, inputs_embeds=inputs_embeds)
        hidden_state = outputs.last_hidden_state

        hidden_state, _ = self.lstm(hidden_state)

        hidden_state = self.dropout(hidden_state)

        logits = self.ffn(hidden_state)

        loss = None
        if labels is not None:
            
            if self.use_crf:
                loss = -1 * self.crf(logits, labels, attention_mask.byte())
            else:            
                loss = self.loss_fn(logits.view(-1, self.nlabels), labels.view(-1))

        return loss, logits

    def predict(self, input_ids, attenton_mask, token_type_ids):

        _, logits = self(
            input_ids,
            attenton_mask,
            token_type_ids,
        )
        
        if self.use_crf:
            predict = self.crf.decode(logits, attenton_mask.byte())
        else:
            predict = logits.argmax(dim=-1)
            predict = predict.cpu().tolist()
        return predict