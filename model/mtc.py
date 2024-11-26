#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch import Tensor
from itertools import chain
from model.layer import FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron
from model.layer import FactorizationMachine, AutomaticFeatureInteraction


class MTCModel(nn.Module):
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
        super().__init__()
        self.model_name = 'mtc'
        self.base_model = base_model  # here use base_model == 'PLE' as an example to show MTC's process
        self.task_types = task_types
        self.embed_output_dim = len(feature_dims) * embed_dim
        self.task_types = task_types
        self.task_num = len(task_types)
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts
        self.experts_dims = experts_dims
        self.layer_num = len(experts_dims)

        # Factorization Machine 一阶+偏置+二阶
        self.embedding = FeaturesEmbedding(feature_dims, embed_dim)
        self.linear = FeaturesLinear(feature_dims)  # 线性+偏置
        self.atten = AutomaticFeatureInteraction(feature_dims, embed_dim, atten_embed_dim=64,
                                                 num_heads=2, num_layers=3, dropout=dropout, has_residual=True)

        self.task_experts = [[0] * self.task_num for _ in range(self.layer_num)]
        self.task_gates = [[0] * self.task_num for _ in range(self.layer_num)]
        self.shared_experts = [0] * self.layer_num
        self.shared_gates = [0] * self.layer_num
        for i in range(self.layer_num):
            input_dim = self.embed_output_dim if i == 0 else experts_dims[i - 1][-1]
            self.shared_experts[i] = nn.ModuleList(
                [MultiLayerPerceptron(input_dim, experts_dims[i], dropout, output_layer=False)
                 for _ in range(self.num_shared_experts)])
            self.shared_gates[i] = nn.Sequential(
                nn.Linear(input_dim, num_shared_experts + self.task_num * num_specific_experts),
                nn.Softmax(dim=1)) if i != self.layer_num-1 else nn.Sequential(
                nn.Linear(input_dim, num_shared_experts), nn.Softmax(dim=1))
            for j in range(self.task_num):
                self.task_experts[i][j] = nn.ModuleList(
                    [MultiLayerPerceptron(input_dim, experts_dims[i], dropout, output_layer=False)
                     for _ in range(self.num_specific_experts)])
                self.task_gates[i][j] = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, num_shared_experts + num_specific_experts), torch.nn.Softmax(dim=1))
            self.task_experts[i] = torch.nn.ModuleList(self.task_experts[i])
            self.task_gates[i] = nn.ModuleList(self.task_gates[i])
        self.task_experts = nn.ModuleList(self.task_experts)
        self.task_gates = nn.ModuleList(self.task_gates)
        self.shared_gates = nn.ModuleList(self.shared_gates)
        self.shared_experts = nn.ModuleList(self.shared_experts)

        self.towers = nn.ModuleList([MultiLayerPerceptron(experts_dims[-1][-1], tower_dims[0], dropout=dropout,
                                                          output_layer=False) for _ in range(self.task_num)])
        self.united_tower = MultiLayerPerceptron(experts_dims[-1][-1], tower_dims[1], dropout=dropout,
                                                 output_layer=False)

        # fm线性+二阶的输出concat=2
        self.output_layers = list()
        for t in self.task_types:
            if t == 'binary':
                self.output_layers.append(nn.Sequential(nn.Linear(2+tower_dims[0][-1], 8), nn.Linear(8, 1), nn.Sigmoid()))
            else:
                self.output_layers.append(nn.Sequential(nn.Linear(2+tower_dims[0][-1], time_dim),
                                                        nn.Linear(time_dim, time_dim)))
        self.output_layers = nn.ModuleList(self.output_layers)
        self.united_output_layers = nn.Sequential(nn.Linear(tower_dims[1][-1], time_dim), nn.Linear(time_dim, time_dim))

    def change_mode(self, mode_str):
        pass

    def united_grad(self, requires_grad=True):
        for p in self.parameters():
            p.requires_grad = not requires_grad
        for p in chain(self.united_tower.parameters(), self.united_output_layers.parameters(),
                       self.shared_gates.parameters()):
            p.requires_grad = requires_grad

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        experts_input = embed_x.view(-1, self.embed_output_dim)

        task_fea = [experts_input for _ in range(self.task_num + 1)]
        for i in range(self.layer_num):
            shared_output = [expert(task_fea[-1]).unsqueeze(1) for expert in self.shared_experts[i]]
            task_output_list = list()
            for j in range(self.task_num):
                task_output = [expert(task_fea[j]).unsqueeze(1) for expert in self.task_experts[i][j]]
                task_output_list.extend(task_output)
                mix_output = torch.cat(task_output + shared_output, dim=1)
                gate_value = self.task_gates[i][j](task_fea[j]).unsqueeze(1)
                task_fea[j] = torch.bmm(gate_value, mix_output).squeeze(1)
            gate_value = self.shared_gates[i](task_fea[-1]).unsqueeze(1) \
                if i != (self.layer_num-1) else self.shared_gates[i](task_fea[-1]).unsqueeze(1)
            mix_output = torch.cat(task_output_list + shared_output, dim=1) \
                if i != self.layer_num - 1 else torch.cat(shared_output, dim=1).detach().clone()  # gradient stop
            task_fea[-1] = torch.bmm(gate_value, mix_output).squeeze(1)
        # size of experts_output:  batch_size * (task_num+1) * expert_num * experts_dims[-1]

        towers_output = [self.towers[i](task_fea[i]).squeeze(1) for i in range(self.task_num)]
        united_tower_output = self.united_tower(task_fea[-1]).squeeze(1)
        x = [torch.cat((self.linear(x), self.atten(embed_x), towers_output[i]), dim=1) for i in range(self.task_num)]
        results = [self.output_layers[i](x[i]).squeeze(1) for i in range(self.task_num)]
        united_result = self.united_output_layers(united_tower_output).squeeze(1)
        final_result = results[-1].detach().clone() - united_result

        return results+[final_result]


class UnitedLoss(nn.Module):
    def __init__(self, list, alpha, beta=1.00, exp_interval=20):
        super(UnitedLoss, self).__init__()
        self.name = 'united'
        losslist = []
        for s in list:
            if s == 'BCELoss':
                losslist.append(torch.nn.BCELoss(reduction='mean'))
            if s == 'CELoss':
                losslist.append(torch.nn.CrossEntropyLoss())
        self.losslist = nn.ModuleList(losslist)
        self.alpha = alpha
        self.beta = beta
        self.exp_interval = exp_interval
        self.mode = 'normal'
        self.loss1, self.loss2, self.loss3 = 0, 0, 0

    def change_mode(self, mode_str):
        self.mode = mode_str

    def forward(self, input: Tensor, target: Tensor):
        input0, input1 = input[0], input[1]  # binary, regression
        input_final = input[-1]
        target0, target1 = target[:, 0], target[:, 1]  # binary, regression
        # 只计算click后time的loss(即有监督的部分)
        input1, target1 = input1[target0 > 0], target1[target0 > 0]
        input_final = input_final[target0 > 0]

        self.loss1 = self.losslist[0](input0, target0.float())
        self.loss2 = self.losslist[1](input1, target1)
        self.loss3 = self.losslist[2](input_final, target1)

        if self.mode != 'united':
            # return self.loss1 + self.alpha * self.loss2
            return self.loss1 + self.alpha * self.loss2 + self.beta * self.loss3
        else:
            return self.beta * self.loss3
        # return self.loss1 + self.alpha * self.loss2 + self.beta * self.loss3
        # return self.alpha * loss1 + loss2
