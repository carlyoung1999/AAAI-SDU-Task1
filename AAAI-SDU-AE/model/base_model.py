'''
Description: 
Author: Li Siheng
Date: 2021-09-28 07:38:36
LastEditTime: 2021-10-27 06:49:38
'''
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, AutoConfig, AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from .adversarial_loss import AdversarialLoss


class BaseAEModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseAEModel')
        
        # * general setting
        parser.add_argument('--eval', action='store_true', default=False)
        parser.add_argument('--checkpoint_path', default=None, type=str)
        parser.add_argument('--seed', default=42, type=int)
        parser.add_argument('--save_dir', default='./save', type=str)
        
        parser.add_argument('--model_name', default='BertLSTMModel', type=str)
        parser.add_argument('--pretrain_model',
                            default='bert-base-uncased',
                            type=str)
        
        parser.add_argument('--bert_lr', default=1e-5, type=float)
        parser.add_argument('--lr', default=1e-5, type=float)
        parser.add_argument('--warmup', default=0.1, type=float)

        # * adversarial training
        parser.add_argument('--adversarial', action='store_true', default=False)
        parser.add_argument('--divergence', default='js', type=str) 
        parser.add_argument('--adv_step_size', default=1e-3, type=float,
                            help="1 (default), perturbation size for adversarial training.")
        parser.add_argument('--adv_alpha', default=1, type=float,
                            help="1 (default), trade off parameter for adversarial training.")
        parser.add_argument('--noise_var', default=1e-5, type=float)
        parser.add_argument('--noise_gamma', default=1e-6, type=float, help="1e-4 (default), eps for adversarial copy training.")
        parser.add_argument('--project_norm_type', default='inf', type=str) 


        # * BertLSTMModel
        parser.add_argument('--use_crf', default=False, action='store_true')
        parser.add_argument('--rnn_size', default=256, type=int)
        parser.add_argument('--rnn_nlayer', default=1, type=int)
        parser.add_argument('--ffn_dropout', default=0.3, type=float)

        return parent_args

    def __init__(self, args, tokenizer):
        super().__init__()

        if isinstance(args, dict):
            args = argparse.Namespace(**args)

        self.save_hyperparameters(args)
        self.tokenizer = tokenizer
        self.config = AutoConfig.from_pretrained(args.pretrain_model)
        
        # * 5 is for label padding
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=5)

        if self.hparams.adversarial:
            self.adv_loss_fn = AdversarialLoss(args) 

    def setup(self, stage) -> None:

        if stage == 'fit':
            train_loader = self.train_dataloader()
            self.total_step = int(self.trainer.max_epochs * len(train_loader) / \
                (self.trainer.gpus * self.trainer.accumulate_grad_batches))
            print('Total training step:', self.total_step)

    def train_inputs(slef, batch):
        return {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'token_type_ids': batch['token_type_ids'],
            'labels': batch['labels']
        }

    def training_step(self, batch, batch_idx):

        inputs = self.train_inputs(batch)
        loss, logits = self(**inputs)

        if self.hparams.adversarial:
            loss = self.adv_loss_fn(self, **inputs)

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

        mask = (batch['labels'] != 5).long()
        ntotal = mask.sum()
        ncorrect = ((logits.argmax(dim=-1) == batch['labels']).long() *
                    mask).sum()
        acc = ncorrect / ntotal

        self.log('valid_loss', loss, on_epoch=True, prog_bar=True)
        self.log("valid_acc", acc, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        paras = list(self.named_parameters())
        paras = [{
            'params':
            [p for n, p in paras if not any(nd in n for nd in no_decay)],
            'weight_decay':
            0.01
        }, {
            'params': [p for n, p in paras if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }]
        optimizer = AdamW(paras, lr=self.hparams.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(self.total_step * self.hparams.warmup),
            self.total_step)

        return [{
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }]
