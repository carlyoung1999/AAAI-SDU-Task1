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
from scorer import *


class BertSpanWCRFModel(BaseAEModel):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(args, tokenizer)

        if isinstance(args, dict):
            args = argparse.Namespace(**args)
        self.args = args
        self.tokenizer = tokenizer
        self.nlabels = args.nlabels

        self.config = AutoConfig.from_pretrained(args.pretrain_model)
        self.bert = AutoModel.from_pretrained(args.pretrain_model)
        self.pretrain_model = args.pretrain_model
        self.bert.resize_token_embeddings(new_num_tokens=len(tokenizer))

        hidden_size = self.config.hidden_size
        self.lstm = nn.LSTM(hidden_size,
                           args.rnn_size,
                           args.rnn_nlayer,
                           bidirectional=True,
                           batch_first=True)

        self.dropout = nn.Dropout(args.ffn_dropout)
        self.span_layer = nn.Linear(2 * args.rnn_size, 4, bias=False)
        self.span_loss_fn = nn.BCEWithLogitsLoss()
        self.backup = {}
        self.logits = None
        self.labels = None

        self.ffn = nn.Linear(2 * args.rnn_size, 6, bias=False)
        self.crf = CRF(6, batch_first=True) # 固定为6的label空间

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, crf_labels=None):
        if "distil" in self.pretrain_model:
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
        else:
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
        hidden_state = outputs.last_hidden_state

        hidden_state, _ = self.lstm(hidden_state)

        hidden_state = self.dropout(hidden_state)  # (bs, seq_len, feat_dim)

        span_logits = self.span_layer(hidden_state)  # (bs, seq_len, 4)
        crf_logits = self.ffn(hidden_state) # (bs, seq_len, 6)

        # self.logits = logits
        # self.labels = labels

        loss = None
        if labels is not None:
            span_loss = self.span_loss_fn(span_logits, labels)
            crf_loss = -1 * self.crf(crf_logits, crf_labels, attention_mask.byte(),reduction="mean")
            loss = span_loss + 0.1 * crf_loss
        return loss, span_logits

    def predict(self, input_ids, attenton_mask, token_type_ids):
        # logits: (bs, seq_len, nlabels)
        _, logits = self(
            input_ids,
            attenton_mask,
            token_type_ids,
        )
        predict = []
        for i in range(logits.shape[-1]):
            if i>=2 and i <=3:
                predict.append( (logits[:,:,i] >= 0.5).int().unsqueeze(-1) )
            else:
                predict.append((logits[:, :, i] >= 0.5).int().unsqueeze(-1))
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

    def train_inputs(slef, batch):
        return {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'token_type_ids': batch['token_type_ids'],
            'labels': batch['labels'],
            'crf_labels': batch['crf_labels'] if 'crf_labels' in batch else None,
        }

    def validation_step(self, batch, batch_idx):
        inputs = self.train_inputs(batch)
        loss, logits = self(**inputs)

        mask = inputs["attention_mask"].unsqueeze(-1).expand(-1, -1, 4)
        ntotal = mask.sum()
        ncorrect = (((logits >= 0.5).long() == batch['labels']).long() * mask).sum()
        acc = ncorrect / ntotal

        self.log('valid_loss', loss, on_epoch=True, prog_bar=True)
        self.log("valid_acc", acc, on_step=True, prog_bar=True)
        return batch

    def validation_epoch_end(self, outputs):
        results = []
        from tqdm import tqdm
        for batch in tqdm(outputs):

            predicts = self.predict(batch['input_ids'],
                                    batch['attention_mask'],
                                    batch['token_type_ids'])

            for idx, predict in enumerate(predicts):
                text = batch['text'][idx]
                offset_mapping = batch['offset_mapping'][idx]

                acronyms, long_forms = self.decode(text, predict,
                                              offset_mapping)

                pred = {
                    'text': batch['text'][idx],
                    'ID': batch['idx'][idx],
                    'acronyms': acronyms,
                    'long-forms': long_forms
                }
                results.append(pred)
        pred_file = os.path.join(self.args.save_path, 'output.json')
        import json
        with open(pred_file, 'w') as f:
            json.dump(results, f, indent=4)
        from argparse import Namespace
        print(f"Use {os.path.join(self.args.data_dir, self.args.test_data)} as test_data")
        eval_args = Namespace(v=True, p=pred_file, g=os.path.join(self.args.data_dir, self.args.test_data))
        p, r, f1 = run_evaluation(eval_args)
        print('Official Scores:')
        print('P: {:.2%}, R: {:.2%}, F1: {:.2%}'.format(p, r, f1))
        #TODO
        self.log('valid_f1', f1, on_epoch=True, prog_bar=True)

    def decode(self, text, labels, offset_mapping):
        acronyms = []
        long_forms = []
        for i in range(len(offset_mapping)):
            # 遍历每一个元素
            start_s_label = labels[i][0]
            end_s_label = labels[i][1]
            start_l_label = labels[i][2]
            end_l_label = labels[i][3]

            # Search 【acronym】
            if start_s_label == 1:
                j = i
                while j < len(offset_mapping):
                    if labels[j][1] == 1:
                        start = offset_mapping[i][0]
                        end = offset_mapping[j][1]
                        acronyms.append([start, end])
                        break
                    elif (j != i) and labels[j][0] == 1:
                        break
                    j = j + 1

            # Search 【long_forms】
            if start_l_label == 1:
                j = i
                while j < len(offset_mapping):
                    if labels[j][3] == 1:
                        start = offset_mapping[i][0]
                        end = offset_mapping[j][1]
                        long_forms.append([start, end])
                        break
                    elif (j != i) and labels[j][2] == 1:
                        break
                    j = j + 1
        return acronyms, long_forms