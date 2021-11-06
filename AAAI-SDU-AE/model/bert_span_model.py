'''
Description:
Author: Li Siheng
Date: 2021-09-28 08:40:34
LastEditTime: 2021-10-12 01:45:34
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


class BertSpanModel(BaseAEModel):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(args, tokenizer)

        if isinstance(args, dict):
            args = argparse.Namespace(**args)

        self.tokenizer = tokenizer
        self.nlabels = args.nlabels

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
        self.span_layer = nn.Linear(2 * args.rnn_size, 4, bias=False)
        self.span_loss = nn.BCEWithLogitsLoss()
        self.backup = {}
        self.logits = None
        self.labels = None

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        hidden_state = outputs.last_hidden_state

        hidden_state, _ = self.lstm(hidden_state)

        hidden_state = self.dropout(hidden_state)  # (bs, seq_len, feat_dim)

        logits = self.span_layer(hidden_state)  # (bs, seq_len, 4)
        self.logits = logits
        self.labels = labels

        loss = None
        if labels is not None:
            loss = self.span_loss(logits, labels)
        return loss, logits

    def predict(self, input_ids, attenton_mask, token_type_ids):
        # logits: (bs, seq_len, nlabels)
        _, logits = self(
            input_ids,
            attenton_mask,
            token_type_ids,
        )
        predict = []
        for i in range(logits.shape[-1]):
            predict.append( (logits[:,:,i] >= 0.5).int().unsqueeze(-1) )
        predict = torch.cat(predict,dim=-1)
        predict = predict.cpu().tolist()
        return predict

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # 只改变word_embedding的梯度
        for name, param in self.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)


    def restore(self, emb_name='word_embeddings'):
        # 只改变word_embedding的梯度
        for name, param in self.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def backward(self, loss, optimizer, optimizer_idx):
        if self.hparams.adversarial:
            loss.backward(retain_graph=True)
        else:
            loss.backward()

    def on_after_backward(self) -> None:
        if self.hparams.adversarial:
            self.attack()
            loss_adv = self.span_loss(self.logits, self.labels)
            loss_adv.backward()
            self.restore()