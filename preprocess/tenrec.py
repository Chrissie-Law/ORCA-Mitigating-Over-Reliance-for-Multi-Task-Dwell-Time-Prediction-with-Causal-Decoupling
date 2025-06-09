#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch.utils.data
import os
import copy
import random
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class TenRec(torch.utils.data.Dataset):
    """
    Tenrec dataset
    Data preparation
    """
    def __init__(self, dataset_path, model_name, task_types, days, print_file=None, time_max=400):
        # 三组的数据存在随机误差，随机读取一组进行模型训练测试
        preprocess_path = dataset_path[:dataset_path.find('.')] + f'_fea{random.randint(0, 2)}.pkl'
        # preprocess_path = dataset_path[:dataset_path.find('.')] + f'_fea0.pkl'
        print(f'dataset path: {preprocess_path}')
        time_bins = [0.1, 1.9, 26.1, 44.1, 72.1, 118.1, 216.1, 314, 2 * time_max - 314]
        if os.path.exists(preprocess_path):
            self.data = pd.read_pickle(preprocess_path)
            if (len(task_types) == 1) and (task_types[0] != 'binary'):
                self.data.drop((self.data.loc[self.data['label_ctr'] == 0]).index, inplace=True)
                self.data = self.data.reset_index(drop=True)
            # self.data = self.data[:20000]

            self.normal_train_features = ['user_id', 'item_id', 'gender', 'age', 'category_second', 'category_first']
            self.labels = ['label_ctr', 'label_time']

            tmp_l = list(range(len(time_bins) - 1))
            # 读取之后再根据time_bin定time_map，一般来说不需要
            self.time_map = dict(zip(tmp_l, [(time_bins[i] + time_bins[i + 1]) / 2 for i in tmp_l]))
            self.model_name = model_name
            self.task_types = task_types

            self.normal_fea_data = self.data[self.normal_train_features]
            self.labels_data = self.data[self.labels]
            self.label0_data = self.data[self.labels[0]]
            self.label1_data = self.data[self.labels[1]]

            self.X = torch.from_numpy(self.normal_fea_data.to_numpy(dtype=np.int64))
            self.Y = torch.from_numpy(self.labels_data.to_numpy(dtype=np.int64))
            self.Y0 = torch.from_numpy(self.label0_data.to_numpy(dtype=np.int64))
            self.Y1 = torch.from_numpy(self.label1_data.to_numpy(dtype=np.int64))            

            self.feature_dims = np.max(self.data[self.normal_train_features], axis=0) + 1  # length of one-hot

            self.print_inf(print_file)
            return

        data = pd.read_csv('data/Tenrec_QK-article_sample.csv')
        column_names = ['user_id', 'item_id', 'click', 'gender', 'age', 'exposure_count',
                        'click_count', 'like_count', 'comment_count', 'read_percentage',
                        'item_score1', 'item_score2', 'category_second', 'category_first',
                        'item_score3', 'read', 'read_time', 'share', 'like', 'follow',
                        'favorite']

        data['label_ctr'] = data['label_ctr'].fillna(0).astype(int)
        data['read_time'] = data['read_time'].fillna(0).astype(float).astype(int)
        data.loc[data['read_time'] >= time_max, 'read_time'] = time_max
        data.loc[data['read_time'] <= 1, 'read_time'] = 1
        tmp_l = list(range(len(time_bins)-1))
        self.time_map = dict(zip(tmp_l, [(time_bins[i] + time_bins[i + 1])/2 for i in tmp_l]))
        data['label_time'] = pd.cut(data['read_time'], bins=time_bins, labels=False)
        self.labels = ['label_ctr', 'label_time']
        self.task_types = task_types
        self.model_name = model_name

        # Select effective features according to "leave-one-out" analysis.
        # Since the TenRec dataset does not have significant title-related features, the fea-level intervention method is not used; also, features such as total clicks and recommendation scores are excluded as they are too strong.
        self.normal_train_features = ['user_id', 'item_id', 'gender', 'age', 'category_second', 'category_first']
        self.total_features = self.normal_train_features

        sparse_features = ['user_id', 'item_id', 'gender', 'category_second', 'category_first']
        dense_features = ['age']
        self.static_features = sparse_features + dense_features  

        data['age'] = data['age'].fillna(data['age'].mean()).astype(int)  # doc_pubtime
        min_age = 8
        data.loc[data['age'] < min_age, 'age'] = min_age
        data['age'] = pd.cut(data['age'], bins=15, labels=False)

        data[self.static_features].fillna('-1', inplace=True)
        for fea in self.static_features:
            lbe = LabelEncoder()
            data[fea] = lbe.fit_transform(data[fea].astype(str))

        self.data = data[self.total_features+self.labels+['read_time']]
        self.data.to_pickle(preprocess_path)
        if (len(task_types) == 1) and (task_types[0] != 'binary'):
            self.data.drop((data.loc[data['label_ctr'] == 0]).index, inplace=True)
            self.data = self.data.reset_index(drop=True)

        self.normal_fea_data = self.data[self.normal_train_features]
        self.labels_data = self.data[self.labels]
        self.label0_data = self.data[self.labels[0]]
        self.label1_data = self.data[self.labels[1]]

        self.X = torch.from_numpy(self.normal_fea_data.to_numpy(dtype=np.int64))
        self.Y = torch.from_numpy(self.labels_data.to_numpy(dtype=np.int64))
        self.Y0 = torch.from_numpy(self.label0_data.to_numpy(dtype=np.int64))
        self.Y1 = torch.from_numpy(self.label1_data.to_numpy(dtype=np.int64))        

        # field dim处理
        self.feature_dims = np.max(data[self.normal_train_features], axis=0) + 1  # length of one-hot

        self.print_inf(print_file)

    def print_inf(self, print_file):
        print('-----------dataset information:-----------')
        print('dataset name: industry')
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

    def change_mode(self, mode_str):
        raise NotImplementedError('TenRec dataset does not support change_mode function, '
                                  'because its features are anonymous.')

    def __getitem__(self, idx):
        x = self.X[idx]  # very fast
        if len(self.task_types) == 2:
            y = self.Y[idx]  # [ctr, time‐bucket]
        elif self.task_types[0] == 'binary':
            y = self.Y0[idx]  # just ctr
        else:
            y = self.Y1[idx]  # just time
        return x, y

    @staticmethod
    def pad_sequence(sequence, maxlen):
        if len(sequence) == maxlen:
            return sequence
        elif len(sequence) < maxlen:
            sequence.extend(['-1'] * (maxlen - len(sequence)))
            return copy.deepcopy(sequence)
        else:
            return copy.deepcopy(sequence[:maxlen])
