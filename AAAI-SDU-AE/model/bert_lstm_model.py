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
from .adversarial_loss import AdversarialLoss
from scorer import *


class BertLSTMModel(BaseAEModel):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(args, tokenizer)

        if isinstance(args, dict):
            args = argparse.Namespace(**args)
        self.args = args
        self.tokenizer = tokenizer
        self.nlabels = args.nlabels
        self.use_crf = args.use_crf

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
        self.ffn = nn.Linear(2 * args.rnn_size, self.nlabels, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

        if self.hparams.adversarial:
            self.adv_loss_fn = AdversarialLoss(args)

        if self.use_crf:
            self.crf = CRF(self.nlabels, batch_first=True)

    def forward(self,
                attention_mask,
                token_type_ids,
                input_ids=None,
                inputs_embeds=None,
                labels=None,
                adversarial_ite=False):
        if "distil" in self.pretrain_model:
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                inputs_embeds=inputs_embeds)
        else:
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                inputs_embeds=inputs_embeds)
        hidden_state = outputs.last_hidden_state

        hidden_state, _ = self.lstm(hidden_state)
        hidden_state = self.dropout(hidden_state)

        logits = self.ffn(hidden_state)

        loss = None
        if labels is not None:

            if self.use_crf:
                loss = -1 * self.crf(logits, labels, attention_mask.byte())
            else:
                loss = self.loss_fn(logits.view(-1, self.nlabels),
                                    labels.view(-1))

            if self.training and self.hparams.adversarial and adversarial_ite is False:

                adv_loss = self.adv_loss_fn(model=self,
                                            logits=logits,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids,
                                            input_ids=input_ids,
                                            labels=labels)
                loss += self.hparams.adv_alpha * adv_loss

        return loss, logits

    def predict(self, input_ids, attention_mask, token_type_ids):

        _, logits = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        if self.use_crf:
            predict = self.crf.decode(logits, attention_mask.byte())
        else:
            predict = logits.argmax(dim=-1)
            predict = predict.cpu().tolist()
        return predict

    def validation_step(self, batch, batch_idx):
        inputs = self.train_inputs(batch)
        loss, logits = self(**inputs)

        mask = (batch['labels'] != 5).long()
        ntotal = mask.sum()
        ncorrect = ((logits.argmax(dim=-1) == batch['labels']).long() *
                    mask).sum()
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
        self.label_list = ['O', 'Bs', 'Is', 'Bl', 'Il', '<ignore>']
        self.label_idx_dict = {
            label: idx
            for idx, label in enumerate(self.label_list)
        }
        for i in range(len(offset_mapping)):
            cur = self.label_list[labels[i]]
            if cur == 'Bs':
                j = i
                while True:
                    j += 1
                    if j == len(labels):
                        j -= 1
                        break
                    next = self.label_list[labels[j]]
                    if next != 'Is':
                        j -= 1
                        break
                start = offset_mapping[i][0]
                end = offset_mapping[j][1]
                if text[start] == ' ':
                    start += 1
                acronyms.append([start, end])
            elif cur == 'Bl':
                j = i
                while True:
                    j += 1
                    if j == len(labels):
                        j -= 1
                        break
                    next = self.label_list[labels[j]]
                    if next != 'Il':
                        j -= 1
                        break
                start = offset_mapping[i][0]
                end = offset_mapping[j][1]
                if text[start] == ' ':
                    start += 1
                long_forms.append([start, end])
        return acronyms, long_forms