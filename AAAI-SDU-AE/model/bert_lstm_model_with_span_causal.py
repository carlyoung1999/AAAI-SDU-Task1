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
from .CausalNormClassifier import Causal_Norm_Classifier


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
        self.clf = Causal_Norm_Classifier(feat_dim=2 * args.rnn_size, num_classes=self.nlabels, use_effect=True,
                                          num_head=4, tau=16.0, alpha=3.0, gamma=0.03125)

        if self.use_crf:
            self.crf = CRF(self.nlabels, batch_first=True)

        if self.use_span == True:
            self.start_s_layer = nn.Linear(2 * args.rnn_size, 1)
            self.end_s_layer = nn.Linear(2 * args.rnn_size, 1)
            self.start_l_layer = nn.Linear(2 * args.rnn_size, 1)
            self.end_l_layer = nn.Linear(2 * args.rnn_size, 1)
            self.start_s_loss = nn.BCEWithLogitsLoss()
            self.end_s_loss = nn.BCEWithLogitsLoss()
            self.start_l_loss = nn.BCEWithLogitsLoss()
            self.end_l_loss = nn.BCEWithLogitsLoss()

        if self.use_effect == True:
            self.mu = 0.9
            # torch.nn.register_buffer() 能够保证该变量在model.save()的时候会跟其他的nn.Parameter对象一样被save。
            self.register_buffer("embed_mean", torch.zeros(int(2 * args.rnn_size)), persistent=True)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None,
                start_s_labels=None, end_s_labels=None,
                start_l_labels=None, end_l_labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        hidden_state = outputs.last_hidden_state

        hidden_state, _ = self.lstm(hidden_state)

        hidden_state = self.dropout(hidden_state)  # (bs, seq_len, feat_dim)

        if self.use_effect == False:
            logits = self.ffn(hidden_state)
        else:
            # hidden_state: (bs, seq_len, feat_dim) --> (bs*seq_len, feat_dim)
            hidden_state_reshaped = hidden_state.reshape(-1, hidden_state.size(2))
            if self.training == True:  # 只在训练的时候记录embedding
                self.embed_mean = self.mu * self.embed_mean + hidden_state_reshaped.detach().mean(0).view(-1)
            logits = self.clf(hidden_state_reshaped, embed=self.embed_mean, stage=self.stage)[
                0]  # (bs*seq_len, num_classes)
        loss = None
        if labels is not None:

            if self.use_crf:
                loss = -1 * self.crf(logits.view(hidden_state.shape[0], hidden_state.shape[1], self.nlabels), labels,
                                     attention_mask.byte(), reduction='mean')
            else:
                loss = self.loss_fn(logits.view(-1, self.nlabels), labels.view(-1))
        if start_s_labels != None:
            start_s_logits = self.start_s_layer(hidden_state)
            end_s_logits = self.end_s_layer(hidden_state)
            start_l_logits = self.start_l_layer(hidden_state)
            end_l_logits = self.end_l_layer(hidden_state)
            loss += 0.8 * self.start_s_loss(start_s_logits.view(-1, 1), start_s_labels.view(-1, 1).float())
            loss += 0.8 * self.end_s_loss(end_s_logits.view(-1, 1), end_s_labels.view(-1, 1).float())
            loss += 0.8 * self.start_l_loss(start_l_logits.view(-1, 1), start_l_labels.view(-1, 1).float())
            loss += 0.8 * self.end_l_loss(end_l_logits.view(-1, 1), end_l_labels.view(-1, 1).float())
            # logits还是用的BiLSTM的logits
        return loss, logits.view(hidden_state.shape[0], hidden_state.shape[1], self.nlabels)

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