#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from model.mtc import MTCModel
from itertools import chain
from model.layer import FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron, FactorizationMachine, Expert, Tower


class MTCFeaModel(MTCModel):
    def __init__(self, base_model, feature_dims, embed_dim, num_shared_experts, num_specific_experts, experts_dims,
                 tower_dims, task_types=['binary', 'regression'], dropout=0.2, time_dim=9):
        """
        :param feature_dims: 每一个sparse feature的one-hot长度，其中同一个field的feature只有一个非0
        :param embed_dim: embedding的大小
        :param num_shared_experts: 多个任务shared的expert个数
        :param num_specific_experts: 每个任务特有的expert个数
        :param experts_dims: expert的各层大小
        :param tower_dims: tower的各层大小
        :param task_types: 任务类型，决定每一个task的loss和output layer，
                           "binary" for binary log loss, "regression" for regression loss
        :param dropout: expert的dropout
        :param time_dim: dwell time的输出分类数目
        """
        super(MTCFeaModel, self).__init__(base_model, feature_dims, embed_dim, num_shared_experts, num_specific_experts,
                                          experts_dims, tower_dims, task_types, dropout, time_dim)
        self.model_name = 'mtc-fea'
        self.mode = 'normal'

    def change_mode(self, mode_str):
        self.mode = mode_str

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        fea_len = x.shape[1]//2
        if self.mode == 'normal':
            x = x[:, :fea_len]
        else:
            x = x[:, fea_len:]
        return super(MTCFeaModel, self).forward(x)

