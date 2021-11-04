'''
Description:
Author: Li Siheng
Date: 2021-10-11 11:00:12
LastEditTime: 2021-10-12 08:17:21
'''
import os
import sys
import json
from pytorch_lightning.callbacks.progress import tqdm
import torch
import argparse
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoTokenizer
from model.data_model import SDUDataModel, SDUDataset
from model.base_model import BaseAEModel
from model.bert_lstm_model import BertLSTMModel
from scorer import *
from argparse import Namespace


def main(args):
    save_path = os.path.join(args.save_dir, args.pretrain_model, args.model_name)

    if args.model_name == 'BertLSTMModel':
        Model = BertLSTMModel
        hyparas = 'use_crf: {} - bert_lr: {} - head_lr: {} - rnn_size: {} - rnn_layer: {} - use_span: {}'.format(
            args.use_crf, args.bert_lr, args.head_lr, args.rnn_size, args.rnn_nlayer, args.use_span)
        save_path = os.path.join(save_path, hyparas)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    seed_everything(args.seed)

    logger = loggers.TensorBoardLogger(save_dir=os.path.join(
        save_path, 'logs/'),
        name='')
    checkpoint = ModelCheckpoint(dirpath=save_path,
                                 save_top_k=1,
                                 monitor='valid_loss',
                                 mode='min',
                                 filename='{epoch:02d}-{valid_loss:.4f}')
    early_stop = EarlyStopping(monitor='valid_loss', mode='min', patience=3)
    trainer = Trainer.from_argparse_args(args,
                                         logger=logger,
                                         callbacks=[checkpoint, early_stop])

    if args.eval is False:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model,
                                                  use_fast=True)

        data_model = SDUDataModel(args, tokenizer)
        model = Model(args, tokenizer)
        trainer.fit(model, data_model)
        tokenizer.save_pretrained(save_path)
        checkpoint_path = checkpoint.best_model_path
    else:
        tokenizer = AutoTokenizer.from_pretrained(save_path)
        data_model = SDUDataModel(args, tokenizer)
        checkpoint_path = os.path.join(args.checkpoint_path)

    # module evaluation
    model = Model.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer)
    evaluation(args, model, data_model, save_path)


def evaluation(args, model, data_model, save_path):
    data_model.setup('test')
    tokenizer = data_model.tokenizer
    test_loader = data_model.test_dataloader()

    device = torch.device('cuda:0')
    model.to(device)
    model.stage = 'test'
    model.eval()

    results = []
    for batch in tqdm(test_loader):

        predicts = model.predict(batch['input_ids'].to(device),
                                 batch['attention_mask'].to(device),
                                 batch['token_type_ids'].to(device))

        for idx, predict in enumerate(predicts):
            text = batch['text'][idx]
            offset_mapping = batch['offset_mapping'][idx]

            acronyms, long_forms = data_model.decode(text, predict, offset_mapping)

            pred = {
                'ID': batch['idx'][idx],
                'acronyms': acronyms,
                'long-forms': long_forms
            }
            results.append(pred)

    pred_file = os.path.join(save_path, 'outputs.json')
    with open(pred_file, 'w') as f:
        json.dump(results, f, indent=4)

    # get [micro, macro]-[precision, recall, f1]
    eval_args = Namespace(v=True, p=pred_file, g=os.path.join(args.data_dir, args.test_data))
    p, r, f1 = run_evaluation(eval_args)
    print('Official Scores:')
    print('P: {:.2%}, R: {:.2%}, F1: {:.2%}'.format(p, r, f1))


if __name__ == '__main__':
    total_parser = argparse.ArgumentParser()

    # * Args for data preprocessing
    total_parser = SDUDataModel.add_data_specific_args(total_parser)

    # * Args for training
    total_parser = Trainer.add_argparse_args(total_parser)

    # * Args for model specific
    total_parser = BaseAEModel.add_model_specific_args(total_parser)

    args = total_parser.parse_args()

    main(args)