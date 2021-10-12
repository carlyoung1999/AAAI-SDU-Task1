'''
Description: 
Author: Li Siheng
Date: 2021-10-10 03:08:16
LastEditTime: 2021-10-12 01:34:43
'''
import argparse
import json
import copy
import os
import torch
from torch.jit import annotate
import torch.nn as nn
import random
import numpy as np
import pytorch_lightning as pl
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


class SDUDataModel(pl.LightningDataModule):
    
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('SDUDataModel')
        parser.add_argument('--data_dir',
                            default='./data/english/scientific',
                            type=str)
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--train_data', default='train.json', type=str)
        parser.add_argument('--valid_data', default='dev.json', type=str)
        parser.add_argument('--test_data', default='dev.json', type=str)
        parser.add_argument('--cached_train_data',
                            default='cached_train_data.pkl',
                            type=str)
        parser.add_argument('--cached_valid_data',
                            default='cached_valid_data.pkl',
                            type=str)
        parser.add_argument('--cached_test_data',
                            default='cached_valid_data.pkl',
                            type=str)
        parser.add_argument('--train_batchsize', default=32, type=int)
        parser.add_argument('--valid_batchsize', default=16, type=int)

        return parent_args
    
    def __init__(self, args, tokenizer):

        super().__init__()
        self.tokenizer = tokenizer
        self.num_workers = args.num_workers
        self.pretrain_model = args.pretrain_model

        # * Notice that we have 6 labels, the last one is for label padding
        self.label_list = ['O', 'Bs', 'Is', 'Bl', 'Il', '<ignore>']
        self.label_idx_dict = {
            label: idx
            for idx, label in enumerate(self.label_list)
        }

        self.cached_train_data_path = os.path.join(args.data_dir,
                                                   args.cached_train_data)
        self.cached_valid_data_path = os.path.join(args.data_dir,
                                                   args.cached_valid_data)
        self.cached_test_data_path = os.path.join(args.data_dir,
                                                  args.cached_test_data)

        self.train_data_path = os.path.join(args.data_dir, args.train_data)
        self.valid_data_path = os.path.join(args.data_dir, args.valid_data)

        self.test_data_path = os.path.join(args.data_dir, args.test_data)

        self.train_batchsize = args.train_batchsize
        self.valid_batchsize = args.valid_batchsize

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == 'fit':
            self.train_data = self.creat_dataset(self.cached_train_data_path,
                                                 self.train_data_path)
            self.valid_data = self.creat_dataset(self.cached_valid_data_path,
                                                 self.valid_data_path)
        if stage == 'test':
            self.test_data = self.creat_dataset(self.cached_test_data_path,
                                                self.test_data_path,
                                                test=True)

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, collate_fn=self.collate_fn, \
            batch_size=self.train_batchsize, num_workers=self.num_workers, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.valid_data, shuffle=False, collate_fn=self.collate_fn, \
            batch_size=self.valid_batchsize, num_workers=self.num_workers, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, collate_fn=self.collate_fn, \
            batch_size=self.valid_batchsize, num_workers=self.num_workers, pin_memory=False)

    def creat_dataset(self, cached_data_path, data_path, test=False):

        if os.path.exists(cached_data_path):
            print('Loading cached dataset...')
            data = torch.load(cached_data_path)
        else:
            print('Preprocess data for SDU...')
            dataset = json.load(open(data_path, 'r'))
            data = []

            total_num = 0
            annotated_correct_num = 0

            for example in dataset:

                total_num += 1
                text = example['text']
                acronyms = example['acronyms']
                long_forms = example['long-forms']

                encoded = self.tokenizer(text, return_offsets_mapping=True)
                input_ids = encoded['input_ids']
                attention_mask = encoded['attention_mask']
                token_type_ids = encoded['token_type_ids']
                offset_mapping = encoded['offset_mapping']
                label = [
                    self.label_idx_dict['O'] for i in range(len(input_ids))
                ]

                # * Construct BIO labels for sequence labelling
                for idx, token_idx in enumerate(input_ids):
                    start = offset_mapping[idx][0]
                    end = offset_mapping[idx][1]
                    if start == end:
                        continue
                    for (acro_start, acro_end) in acronyms:

                        if start == acro_start or start == acro_start - 1 and text[
                                start] == ' ':
                            label[idx] = self.label_idx_dict['Bs']
                        elif start > acro_start and end <= acro_end:
                            label[idx] = self.label_idx_dict['Is']
                    for (long_start, long_end) in long_forms:

                        if start == long_start or start == long_start - 1 and text[
                                start] == ' ':
                            label[idx] = self.label_idx_dict['Bl']
                        elif start > long_start and end <= long_end:
                            label[idx] = self.label_idx_dict['Il']

                # * Notice that we must confirm that we can acquire ground-truth
                # * acronyms and long-forms with ground-truth labels
                decode_acronyms, decode_long_forms = self.decode(
                    text, label, offset_mapping)

                if sorted(acronyms) == sorted(decode_acronyms) and sorted(
                        long_forms) == sorted(decode_long_forms):
                    annotated_correct_num += 1
                else:
                    pass
                    # * Have a look at the error annotation
                    # print(text)
                    # print('Gronund-truth')
                    # print(acronyms)
                    # print(long_forms)
                    # print(encoded)

                    # print('Decode')
                    # print(decode_acronyms)
                    # print(decode_long_forms)

                example = {
                    'idx': example['ID'],
                    'text': text,
                    'offset_mapping': offset_mapping,
                    'input_ids': torch.LongTensor(input_ids),
                    'attention_mask': torch.LongTensor(attention_mask),
                    'token_type_ids': torch.LongTensor(token_type_ids),
                    'labels': torch.LongTensor(label),
                }
                data.append(example)

            output = f'In {data_path}, there are {total_num} instances and {annotated_correct_num} is right, the ration is {annotated_correct_num/total_num}'
            print(output)

            data = SDUDataset(data)
            torch.save(data, cached_data_path)

        return data

    def decode(self, text, labels, offset_mapping):
        """This function used for generating acronyms and long_forms given the BIO label

        Args:
            text (str): text information
            labels (list[int]): BIO labels
            offset_mapping (list[(int, int)]): len * [start, end], each represents the position of token in labels

        Returns:        
            acronyms (list[[int, int]]): The detected acronyms position in text
            long-forms (list[[int, int]]): The detected long-forms position in text
        """
        acronyms = []
        long_forms = []
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

    def collate_fn(self, batch):

        batch_data = {}
        for key in batch[0]:
            batch_data[key] = [example[key] for example in batch]

        input_ids = batch_data['input_ids']
        attention_mask = batch_data['attention_mask']
        token_type_ids = batch_data['token_type_ids']
        labels = batch_data['labels']

        input_ids = nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)

        attention_mask = nn.utils.rnn.pad_sequence(attention_mask,
                                                   batch_first=True,
                                                   padding_value=0)
        token_type_ids = nn.utils.rnn.pad_sequence(token_type_ids,
                                                   batch_first=True,
                                                   padding_value=0)
        labels = nn.utils.rnn.pad_sequence(labels,
                                           batch_first=True,
                                           padding_value=5)
        batch_data = {
            'idx': batch_data['idx'],
            'text': batch_data['text'],
            'offset_mapping': batch_data['offset_mapping'],
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels,
        }

        return batch_data


class SDUDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


if __name__ == '__main__':

    total_parser = argparse.ArgumentParser()
    parser = total_parser.add_argument_group('Program Arguments')
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--data_dir',
                        default='./data/english/scientific',
                        type=str)
    parser.add_argument('--save_dir', default='./save', type=str)
    parser.add_argument('--model_name', default='BertLSTMModel', type=str)
    parser.add_argument('--checkpoint_path', default=None, type=str)
    parser.add_argument('--train_data', default='train.json', type=str)
    parser.add_argument('--valid_data', default='dev.json', type=str)
    parser.add_argument('--test_data', default='dev.json', type=str)
    parser.add_argument('--cached_train_data',
                        default='cached_train_data.pkl',
                        type=str)
    parser.add_argument('--cached_valid_data',
                        default='cached_valid_data.pkl',
                        type=str)
    parser.add_argument('--cached_test_data',
                        default='cached_valid_data.pkl',
                        type=str)
    parser.add_argument('--train_batchsize', default=32, type=int)
    parser.add_argument('--valid_batchsize', default=16, type=int)

    parser.add_argument('--pretrain_model',
                        default='bert-base-uncased',
                        type=str)

    args = total_parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model,
                                              use_fast=True)

    sdu_data = SDUDataModel(args, tokenizer)

    sdu_data.setup('fit')

    val_dataloader = sdu_data.val_dataloader()

    batch = next(iter(val_dataloader))

    print(batch)