'''
Description: 
Author: Li Siheng
Date: 2021-10-11 11:00:12
LastEditTime: 2021-10-12 01:33:58
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


def main(args):

    save_path = os.path.join(args.save_dir, args.model_name)

    if args.model_name == 'BertLSTMModel':
        Model = BertLSTMModel
        hyparas = 'use_crf: {} - lr: {} - rnn_size: {} - rnn_layer: {}'.format(
            args.use_crf, args.lr, args.rnn_size, args.rnn_nlayer)
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
        checkpoint_path = os.path.join(save_path, args.checkpoint_path)

    # module evaluation
    model = Model.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer)
    evaluation(args, model, data_model, save_path)


def evaluation(args, model, data_model, save_path):

    data_model.setup('test')
    tokenizer = data_model.tokenizer
    test_loader = data_model.test_dataloader()

    device = torch.device('cuda:0')
    model.to(device)
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

    with open(os.path.join(save_path, 'outputs.json'), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    total_parser = argparse.ArgumentParser()

    # * args for data preprocessing
    total_parser = SDUDataModel.add_data_specific_args(total_parser)
    
    # * args for training
    total_parser = Trainer.add_argparse_args(total_parser)

    # * args for model specific
    total_parser = BaseAEModel.add_model_specific_args(total_parser)
    
    # * args for general setting
    parser = total_parser.add_argument_group('Program Arguments')
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--save_dir', default='./save', type=str)
    parser.add_argument('--model_name', default='BertLSTMModel', type=str)
    parser.add_argument('--pretrain_model',
                        default='bert-base-uncased',
                        type=str)
    parser.add_argument('--nlabels', default=6, type=int)

    print(parser)
    args = total_parser.parse_args()

    main(args)