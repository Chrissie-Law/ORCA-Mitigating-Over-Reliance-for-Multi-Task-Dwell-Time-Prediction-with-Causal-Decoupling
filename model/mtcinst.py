#!/usr/bin/env python
# -*- coding: utf-8 -*-
from model.mtc import MTCModel
from model.mtcfea import MTCFeaModel
from model.mtcctr import MTCCtrModel, MTCFeaCtrModel
from torch import nn, ones
from torch import Tensor
import torch.nn.functional as F


class MTCInstModel(MTCModel):
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
        super(MTCInstModel, self).__init__(base_model, feature_dims, embed_dim, num_shared_experts,
                                           num_specific_experts, experts_dims, tower_dims, task_types, dropout, time_dim)
        self.model_name = 'mtc-inst'


class MTCFeaInstModel(MTCFeaModel):
    def __init__(self, base_model, feature_dims, embed_dim, num_shared_experts, num_specific_experts, experts_dims,
                 tower_dims, task_types=['binary', 'regression'], dropout=0.2, time_dim=9):
        super(MTCFeaInstModel, self).__init__(base_model, feature_dims, embed_dim, num_shared_experts,
                                              num_specific_experts, experts_dims, tower_dims, task_types, dropout, time_dim)
        self.model_name = 'mtc-fea-inst'


class MTCCtrInstModel(MTCCtrModel):
    def __init__(self, base_model, feature_dims, embed_dim, num_shared_experts, num_specific_experts, experts_dims,
                 tower_dims, task_types=['binary', 'regression'], dropout=0.2, time_dim=9, guide_dim=0):
        super(MTCCtrInstModel, self).__init__(base_model, feature_dims, embed_dim, num_shared_experts,
                                              num_specific_experts, experts_dims, tower_dims, task_types,
                                              dropout, time_dim, guide_dim)
        if guide_dim > 0:
            self.model_name = 'mtc-ctr-inst-guide'  # 通过一层神经网络再输入DT塔
        elif guide_dim == 0:
            self.model_name = 'mtc-ctr-inst'  # 直接拼接输入DT塔
        else:
            self.model_name = 'mtc-ctr-inst-gate'  # 作为gate控制DT塔的输出


class MTCFeaCtrInstModel(MTCFeaCtrModel):
    def __init__(self, base_model, feature_dims, embed_dim, num_shared_experts, num_specific_experts, experts_dims,
                 tower_dims, task_types=['binary', 'regression'], dropout=0.2, time_dim=9, guide_dim=0):
        super(MTCFeaCtrInstModel, self).__init__(base_model, feature_dims, embed_dim, num_shared_experts,
                                                 num_specific_experts, experts_dims, tower_dims, task_types,
                                                 dropout, time_dim, guide_dim)
        if guide_dim > 0:
            self.model_name = 'mtc-fea-ctr-inst-guide'  # 通过一层神经网络再输入DT塔
        elif guide_dim == 0:
            self.model_name = 'mtc-fea-ctr-inst'  # 直接拼接输入DT塔
        else:
            self.model_name = 'mtc-fea-ctr-inst-gate'  # 作为gate控制DT塔的输出


class UnitedWeightedLossMultiLoss1Loss2(nn.Module):
    def __init__(self, list, alpha, beta=1.01, exp_interval=20, offset=0, ctr_weight=1, dt_weight=1):
        super(UnitedWeightedLossMultiLoss1Loss2, self).__init__()
        self.name = 'weighted'
        losslist = []
        for s in list:
            if s == 'BCELoss':
                losslist.append(nn.BCELoss(reduction='mean'))
            if s == 'CELoss':
                losslist.append(nn.CrossEntropyLoss(reduction='mean'))
            if s == 'CELoss_none':
                losslist.append(nn.CrossEntropyLoss(reduction='none'))

        self.losslist = nn.ModuleList(losslist)
        self.alpha = alpha
        self.beta = beta
        self.exp_interval = exp_interval
        self.mode = 'normal'
        self.loss1, self.loss2, self.loss3 = 0, 0, 0
        self.weight_loss1 = nn.BCELoss(reduction='none')
        self.weight_loss2 = nn.CrossEntropyLoss(reduction='none')
        self.sample_weight = 0
        self.offset = offset
        self.ctr_weight = ctr_weight
        self.dt_weight = dt_weight

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

        if self.mode != 'united':
            click_input0, click_target0 = input0[target0 > 0].detach().clone(), target0[target0 > 0].detach().clone()
            self.sample_weight = F.relu(self.weight_loss1(click_input0, click_target0.float()).pow(
                self.ctr_weight) * self.weight_loss2(input1.detach().clone(), target1.detach().clone()).pow(
                self.dt_weight) + self.offset)
            # 用正常结果的Loss1+Loss2做权重
            if self.sample_weight.sum() > 1e-2:
                self.sample_weight = (self.sample_weight / self.sample_weight.sum() * click_input0.shape[0]).detach()
            else:
                self.sample_weight = ones(self.sample_weight.shape())
            return self.loss1 + self.alpha * self.loss2
        else:  # united: 只优化模型Tower C'
            self.loss3 = self.losslist[2](input_final, target1) * self.sample_weight
            self.loss3 = self.loss3.mean()
            return self.beta * self.loss3


class UnitedWeightedLoss(nn.Module):
    def __init__(self, list, alpha, beta=1.01, exp_interval=20):
        super(UnitedWeightedLoss, self).__init__()
        losslist = []
        for s in list:
            if s == 'BCELoss':
                losslist.append(nn.BCELoss(reduction='mean'))
            if s == 'CELoss':
                losslist.append(nn.CrossEntropyLoss(reduction='mean'))
            if s == 'CELoss_none':
                losslist.append(nn.CrossEntropyLoss(reduction='none'))
        self.losslist = nn.ModuleList(losslist)
        self.alpha = alpha
        self.beta = beta
        self.exp_interval = exp_interval
        self.mode = 'normal'
        self.loss1, self.loss2, self.loss3 = 0, 0, 0
        self.weight_loss = nn.CrossEntropyLoss(reduction='none')
        self.sample_weight = 0

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
        if self.mode != 'united':
            self.sample_weight = self.weight_loss(input1.detach(), target1.detach()).clone()  # 用正常结果的Loss做权重
            self.sample_weight = (self.sample_weight / self.sample_weight.sum() * target1.shape[0]).detach()
            return self.loss1 + self.alpha * self.loss2
        else:  # united: 只优化Tower C'
            self.loss3 = self.losslist[2](input_final, target1) * self.sample_weight
            self.loss3 = self.loss3.mean()
            return self.beta * self.loss3


class UnitedWeightedLossSumOthers(nn.Module):
    def __init__(self, list, alpha, beta=1.01, exp_interval=20, offset = 0):
        super(UnitedWeightedLossSumOthers, self).__init__()
        losslist = []
        for s in list:
            if s == 'BCELoss':
                losslist.append(nn.BCELoss(reduction='mean'))
            if s == 'CELoss':
                losslist.append(nn.CrossEntropyLoss(reduction='mean'))
            if s == 'CELoss_none':
                losslist.append(nn.CrossEntropyLoss(reduction='none'))
        self.losslist = nn.ModuleList(losslist)
        self.alpha = alpha
        self.beta = beta
        self.exp_interval = exp_interval
        self.mode = 'normal'
        self.loss1, self.loss2, self.loss3 = 0, 0, 0
        self.sample_weight = 0
        self.offset = offset

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

        if self.mode != 'united':
            # DT非真实桶的概率相加 + 偏移量
            self.sample_weight = 1 - input1[:, target1].detach().clone() + self.offset
            self.sample_weight = self.sample_weight / self.sample_weight.sum() * target1.shape[0]
            return self.loss1 + self.alpha * self.loss2
        else:  # united: 只优化Tower C'
            self.loss3 = self.losslist[-1](input_final, target1) * self.sample_weight
            self.loss3 = self.loss3.mean()
            return self.beta * self.loss3


class UnitedWeightedLossSumCtr(nn.Module):
    def __init__(self, list, alpha, beta=1.01, exp_interval=20, offset=0, ctr_weight=1):
        super(UnitedWeightedLossSumCtr, self).__init__()
        losslist = []
        for s in list:
            if s == 'BCELoss':
                losslist.append(nn.BCELoss(reduction='mean'))
            if s == 'CELoss':
                losslist.append(nn.CrossEntropyLoss(reduction='mean'))
            if s == 'CELoss_none':
                losslist.append(nn.CrossEntropyLoss(reduction='none'))
        self.losslist = nn.ModuleList(losslist)
        self.alpha = alpha
        self.beta = beta
        self.exp_interval = exp_interval
        self.mode = 'normal'
        self.loss1, self.loss2, self.loss3 = 0, 0, 0
        self.sample_weight = 0
        self.offset = offset
        self.ctr_weight = ctr_weight
        self.weight_loss = nn.BCELoss(reduction='none')

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

        if self.mode != 'united':
            # DT非真实桶的概率相加 + CTR概率 + 偏移量
            click_input0, click_target0 = input0[target0 > 0].detach().clone(), target0[target0 > 0].detach().clone()
            self.sample_weight = 1 - input1[:, target1].detach().clone() + self.offset + \
                                 self.ctr_weight * self.weight_loss(click_input0, click_target0.float())
            self.sample_weight = self.sample_weight / self.sample_weight.sum() * target1.shape[0]
            return self.loss1 + self.alpha * self.loss2
        else:  # united: 只优化模型Tower C'
            self.loss3 = self.losslist[-1](input_final, target1) * self.sample_weight
            self.loss3 = self.loss3.mean()
            return self.beta * self.loss3


class UnitedWeightedLossArgmax(nn.Module):
    def __init__(self, list, alpha, beta=1.01, exp_interval=20):
        super(UnitedWeightedLossArgmax, self).__init__()
        losslist = []
        for s in list:
            if s == 'BCELoss':
                losslist.append(nn.BCELoss(reduction='mean'))
            if s == 'CELoss':
                losslist.append(nn.CrossEntropyLoss(reduction='none'))
        self.losslist = nn.ModuleList(losslist)
        self.alpha = alpha
        self.beta = beta
        self.exp_interval = exp_interval
        self.mode = 'normal'
        self.loss1, self.loss2, self.loss3 = 0, 0, 0
        self.weight_loss = nn.L1Loss(reduction='none')

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

        if self.mode != 'united':
            self.sample_weight = self.weight_loss(input1.argmax(dim=1), target1).detach().clone() + 1 # 避免出现权重0
            self.sample_weight = self.sample_weight / self.sample_weight.sum() * target1.shape[0]
            self.loss2 = self.loss2.mean()
            return self.loss1 + self.alpha * self.loss2
        else:  # united: 只优化Tower C'
            self.loss3 = self.losslist[2](input_final, target1) * self.sample_weight
            self.loss2 = self.loss2.mean()
            self.loss3 = self.loss3.mean()
            return self.beta * self.loss3

