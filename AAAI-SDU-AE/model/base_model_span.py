'''
Description:
Author: Li Siheng
Date: 2021-09-28 07:38:36
LastEditTime: 2021-10-12 08:15:09
'''
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, AutoConfig, AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union


class BaseAEModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseAEModel')

        # * Args for general setting
        parser.add_argument('--eval', action='store_true', default=False)
        parser.add_argument('--checkpoint_path', default=None, type=str)
        parser.add_argument('--seed', default=42, type=int)
        parser.add_argument('--save_dir', default='./save', type=str)

        parser.add_argument('--model_name', default='BertLSTMModel', type=str)
        parser.add_argument('--pretrain_model',
                            default='bert-base-uncased',
                            type=str)

        parser.add_argument('--bert_lr', default=1e-5, type=float)
        parser.add_argument('--head_lr', default=1e-4, type=float)
        parser.add_argument('--warmup', default=0.1, type=float)

        # * Args for BertLSTMModel
        parser.add_argument('--use_crf', default=False, action='store_true')
        parser.add_argument('--use_span', default=False, action='store_true')
        parser.add_argument('--rnn_size', default=256, type=int)
        parser.add_argument('--rnn_nlayer', default=1, type=int)
        parser.add_argument('--ffn_dropout', default=0.3, type=float)

        parser.add_argument('--use_effect', default=False, action='store_true')

        return parent_args

    def __init__(self, args, tokenizer):
        super().__init__()

        if isinstance(args, dict):
            args = argparse.Namespace(**args)

        self.save_hyperparameters(args)  # save arguments to hparams attribute
        self.tokenizer = tokenizer
        self.config = AutoConfig.from_pretrained(args.pretrain_model)

        # * 5 is for label padding
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=5)  # This setting if fucking good, ignore the influence of 'label padding'

        self.use_span = args.use_span
        self.use_crf = args.use_crf
        self.use_effect = args.use_effect
        self.stage = 'None'

    def setup(self, stage) -> None:

        if stage == 'fit':
            train_loader = self.train_dataloader()  # just get the dataloader
            self.total_step = int(self.trainer.max_epochs * len(train_loader) / \
                                  (self.trainer.gpus * self.trainer.accumulate_grad_batches))
            print('Total training step:', self.total_step)
            self.stage = 'fit'
        else:
            self.stage = 'test'

    def train_inputs(self, batch):
        return {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'token_type_ids': batch['token_type_ids'],
            'labels': batch['labels']
        }

    def training_step(self, batch, batch_idx):

        inputs = self.train_inputs(batch)
        loss, logits = self(**inputs)

        if self.use_span == True:
            # 只统计非mask的token的正确率
            # attention_mask: (bs, seq_len)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(-1, -1, 4)
            ntotal = mask.sum()
            ncorrect = (((logits >= 0.5).long() == batch['labels']).long() * mask).sum()
        if self.use_crf == True:
            mask = (batch['labels'] != 5).long()
            ntotal = mask.sum()
            ncorrect = ((logits.argmax(dim=-1) == batch['labels']).long() *
                        mask).sum()
        acc = ncorrect / ntotal

        self.log('train_loss', loss, on_step=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        inputs = self.train_inputs(batch)
        loss, logits = self(**inputs)

        if self.use_span == True:
            # 只统计非mask的token的正确率
            # attention_mask: (bs, seq_len)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(-1, -1, 4)
            ntotal = mask.sum()
            ncorrect = (((logits >= 0.5).long() == batch['labels']).long() * mask).sum()
            acc = ncorrect / ntotal

            # 直接进行predict
            # predict = []
            # for i in range(logits.shape[-1]):
            #     predict.append((logits[:, :, i] >= 0.5).int().unsqueeze(-1))
            # predict = torch.cat(predict, dim=-1)
            # # predict: (bs, seq_len, nlabels)
            # predict = predict.cpu().tolist()

        if self.use_crf == True:
            mask = (batch['labels'] != 5).long()
            ntotal = mask.sum()
            ncorrect = ((logits.argmax(dim=-1) == batch['labels']).long() *
                        mask).sum()  # 使用的是token级别的评测方法，该方法带来的结果可能就是虽然acc很高，但是其实是由于‘O’带来的影响
            acc = ncorrect / ntotal

        self.log('valid_loss', loss, on_epoch=True, prog_bar=True)
        self.log("valid_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_paras = list(self.bert.named_parameters())
        bert_paras = \
            [
                {'params': [p for n, p in bert_paras if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
                 'lr': self.hparams.bert_lr},
                {'params': [p for n, p in bert_paras if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': self.hparams.bert_lr}
            ]

        named_paras = list(self.named_parameters())
        head_paras = [
            {'params': [p for n, p in named_paras if 'bert' not in n], 'lr': self.hparams.head_lr}
        ]

        paras = bert_paras + head_paras

        optimizer = AdamW(paras, lr=self.hparams.head_lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(self.total_step * self.hparams.warmup),
                                                    self.total_step)

        return [
            {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        ]
