#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch.utils.data
import os
import copy
import random
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class IndustryDataset(torch.utils.data.Dataset):
    """
    Data preparation
    :param dataset_path: Industry dataset path
    Note: Due to confidentiality requirements, specific details of industrial dataset processing are not disclosed.
    """
    def __init__(self, dataset_path, model_name, task_types, days, print_file=None, time_max=400):
        preprocess_path = dataset_path[:dataset_path.find('.')] + f'_{len(days)}day_blockfea{random.randint(0, 2)}.pkl'
        print(f'dataset path: {preprocess_path}')
        # 对dwell time进行分桶：0.1是为了处理0值，1.9是为了处理1值（即打开后马上就关闭），后续按照log值均分
        time_bins = [0.1, 1.9, 26.1, 44.1, 72.1, 118.1, 216.1, 314, 2 * time_max - 314]
        if os.path.exists(preprocess_path):
            self.data = pd.read_pickle(preprocess_path)
            if (len(task_types) == 1) and (task_types[0] != 'binary'):
                self.data.drop((self.data.loc[self.data['label_ctr'] == 0]).index, inplace=True)
                self.data = self.data.reset_index(drop=True)

            self.normal_train_features = ['uin', 'hashed_docid', 'net_type', 'feeds_show_type', 'search_time',
                                          'title_tag_hash0', 'title_tag_hash1', 'title_tag_hash2',
                                          'category', 'subcategory', 'doc_pubtime', 'data_source']
            self.block_content_features = ['uin', 'hashed_docid', 'net_type', 'feeds_show_type', 'search_time',
                                           'title_tag_hash0', 'title_tag_hash1', 'title_tag_hash2',  'category_block',
                                           'subcategory_block', 'doc_pubtime_block', 'data_source_block']
            self.dynamic_features = ['title_tag_hash0', 'title_tag_hash1', 'title_tag_hash2']  # max len=3
            self.labels = ['label_ctr', 'label_time']

            tmp_l = list(range(len(time_bins) - 1))
            # 读取之后再根据time_bin定label，一般来说不需要
            self.data['label_time'] = pd.cut(self.data['read_time'], bins=time_bins, labels=False)
            self.time_map = dict(zip(tmp_l, [(time_bins[i] + time_bins[i + 1]) / 2 for i in tmp_l]))
            self.model_name = model_name
            self.task_types = task_types

            self.normal_and_block_fea_data = self.data[self.normal_train_features + self.block_content_features]
            self.normal_fea_data = self.data[self.normal_train_features]
            self.labels_data = self.data[self.labels]
            self.label0_data = self.data[self.labels[0]]
            self.label1_data = self.data[self.labels[1]]

            self.feature_dims = np.max(self.data[self.normal_train_features], axis=0) + 1  # length of one-hot
            self.feature_dims[self.dynamic_features[-1]] = self.feature_dims[self.dynamic_features].max()
            self.feature_dims[self.dynamic_features[:-1]] = 0

            self.print_inf(print_file)
            return
        pass
        # Due to confidentiality requirements, specific details of industrial dataset processing are not disclosed.

    def print_inf(self, print_file):
        print('-----------dataset information:-----------')
        print(f'dataset name: industry')
        print(f'length of data: {self.data.shape[0]}')
        print(f'feature size: {len(self.normal_train_features)}')
        print(f'feature dims: {self.feature_dims.sum()}')
        print(f'model name: {self.model_name}')
        print(f'target(s): {self.task_types}')
        print(f'dwell time output dim: {len(self.time_map)}')
        total_ctr = self.data['label_ctr'].sum()/self.data['label_ctr'].count()
        print(f'total ctr: {total_ctr}')
        print('------------------------------------------')

        if print_file is None:
            return
        print('-----------dataset information:-----------', file=print_file)
        print('dataset name: industry', file=print_file)
        print('dataset name: industry', file=print_file)
        print(f'length of data: {self.data.shape[0]}', file=print_file)
        print(f'feature size: {len(self.normal_train_features)}', file=print_file)
        print(f'feature dims: {self.feature_dims.sum()}', file=print_file)
        print(f'model name: {self.model_name}', file=print_file)
        print(f'target(s): {self.task_types}', file=print_file)
        print(f'dwell time output dim: {len(self.time_map)}', file=print_file)
        print(f'total ctr: {total_ctr}', file=print_file)
        print('------------------------------------------', file=print_file)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):  # 基于索引进行后续操作，所以需要重定义
        if len(self.task_types) == 2:
            if 'fea' in self.model_name:  # or 'ctr' in self.model_name:
                return self.normal_and_block_fea_data.iloc[index].values, \
                       self.labels_data.iloc[index].values  # block feature加入训练
            else:
                return self.normal_fea_data.iloc[index].values, \
                       self.labels_data.iloc[index].values  # 多目标
        elif self.task_types[0] == 'binary':
            return self.normal_fea_data.iloc[index].values, \
                   self.label0_data.iloc[index]
        else:
            return self.normal_fea_data.iloc[index].values, \
                   self.label1_data.iloc[index]

    @staticmethod
    def pad_sequence(sequence, maxlen):
        if len(sequence) == maxlen:
            return sequence
        elif len(sequence) < maxlen:
            sequence.extend(['-1'] * (maxlen - len(sequence)))
            return copy.deepcopy(sequence)
        else:
            return copy.deepcopy(sequence[:maxlen])



