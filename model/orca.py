#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn, ones
import torch.nn.functional as F
from itertools import chain
from model.mmoe import MMoE_Experts
from model.ple_afi import PLE_Experts
from model.layer import FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron
from model.layer import FactorizationMachine, AutomaticFeatureInteraction


class ORCA(nn.Module):
    def __init__(self, feature_dims, embed_dim, num_shared_experts, num_specific_experts, experts_dims,
                 tower_dims, task_types=['binary', 'regression'], dropout=0.2, time_dim=9,
                 base_model='mmoe', use_inter=False):
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
        super(ORCA, self).__init__()
        self.model_name = 'orca'
        self.task_types = task_types
        self.base_model = base_model
        self.embed_output_dim = len(feature_dims) * embed_dim
        self.task_types = task_types
        self.task_num = len(task_types)
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts
        self.layer_num = len(experts_dims)
        self.use_inter = use_inter

        # Factorization Machine
        self.embedding = FeaturesEmbedding(feature_dims, embed_dim)
        self.linear = FeaturesLinear(feature_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

        # Attention layer for task-interaction
        self.atten = AutomaticFeatureInteraction([0,1], experts_dims[-1][-1], atten_embed_dim=experts_dims[-1][-1],
                                                 num_heads=2, num_layers=1, dropout=dropout, has_residual=True,
                                                 output_dim=experts_dims[-1][-1])

        self.towers = list()
        for t in self.task_types:
            if t == 'binary':
                self.towers.append(MultiLayerPerceptron(experts_dims[-1][-1], tower_dims[0],
                                                        dropout=dropout, output_layer=False))
            else:
                self.towers.append(MultiLayerPerceptron(experts_dims[-1][-1], tower_dims[0],
                                                        dropout=dropout, output_layer=False))
                if use_inter:
                    self.united_tower = MultiLayerPerceptron(experts_dims[-1][-1] * 2, tower_dims[1],
                                                             dropout=dropout, output_layer=False)
                else:
                    self.united_tower = MultiLayerPerceptron(experts_dims[-1][-1], tower_dims[1],
                                                             dropout=dropout, output_layer=False)
        self.towers = nn.ModuleList(self.towers)

        self.output_layers = list()
        for t in self.task_types:
            if t == 'binary':
                self.output_layers.append(
                    nn.Sequential(nn.Linear(2 + tower_dims[0][-1], 8),
                                  nn.ReLU(),
                                  nn.Linear(8, 1),
                                  nn.Sigmoid()))
            else:
                self.output_layers.append(nn.Sequential(nn.Linear(2 + tower_dims[1][-1], time_dim),
                                                        nn.ReLU(),
                                                        nn.Linear(time_dim, time_dim)))
        self.output_layers = nn.ModuleList(self.output_layers)

        self.united_output_layers = nn.Sequential(nn.Linear(2 + tower_dims[1][-1], time_dim),
                                                  nn.ReLU(),
                                                  nn.Linear(time_dim, time_dim))

        if base_model == 'mmoe':
            if type(experts_dims[0]) is int:
                mmoe_experts_dims = experts_dims
            else:
                mmoe_experts_dims = list(chain(*experts_dims))

            self.experts = MMoE_Experts(self.embed_output_dim, mmoe_experts_dims, num_shared_experts,
                                        self.task_num, dropout)
        elif base_model == 'ple':
            self.experts = PLE_Experts(self.embed_output_dim, num_shared_experts, num_specific_experts,
                                       experts_dims, self.task_num, dropout)
        elif base_model == 'metabalance':
            self.experts = PLE_Experts(self.embed_output_dim, num_shared_experts, num_specific_experts,
                                       experts_dims, self.task_num, dropout)
            self.sharedLayerParameters = list(self.embedding.parameters()) + list(self.linear.parameters()) \
                                         + list(self.experts.parameters()) + list(self.fm.parameters())
            self.taskLayerParameters = list(self.towers.parameters()) + list(self.output_layers.parameters()) \
                                       + list(self.united_tower.parameters()) + list(self.atten.parameters()) \
                                       + list(self.united_output_layers.parameters())
        else:
            raise ValueError("base_model not supported: {}".format(base_model))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)  # shape: batch_size * num_fields * embed_dim
        experts_input = embed_x.view(-1, self.embed_output_dim)
        task_fea = self.experts(experts_input)  # shape: batch_size * experts_dims[-1]
        if self.base_model == 'mmoe':
            task_fea[-1] = task_fea[-1].detach()  # detach the last task's output for CTR

        tower_output0 = self.towers[0](task_fea[0])  # ctr tower output
        tower_output1 = self.towers[1](task_fea[1])
        if self.use_inter:
            inter_task_fea = self.atten(torch.stack(task_fea[:2], dim=1).detach())
            united_tower_input = torch.cat((inter_task_fea, task_fea[-1]), dim=1)
        else:
            united_tower_input = task_fea[-1]
        united_tower_output = self.united_tower(united_tower_input)  # united tower output
        towers_output = [tower_output0, tower_output1]

        other_part = torch.cat((self.linear(x), self.fm(embed_x)), dim=1)
        results = [self.output_layers[i](torch.cat((other_part, towers_output[i]), dim=1)).squeeze(1)
                   for i in range(self.task_num)]
        united_result = self.united_output_layers(torch.cat((other_part.detach(), united_tower_output), dim=1)).squeeze(1)
        final_result = results[-1].detach().clone() - united_result

        return results+[final_result]


class Causal_Weighted_Loss(nn.Module):
    def __init__(self, alpha=1, beta=1, offset=0, ctr_weight=1, dt_weight=1,
                 is_calc_inst_weight=False, ips_mode='ciips',
                 lambda_cap: float = 10.0):
        super(Causal_Weighted_Loss, self).__init__()
        self.name = 'causal_weighted'
        self.is_calc_inst_weight, self.ips_mode = is_calc_inst_weight, ips_mode
        self.lambda_cap = lambda_cap
        if ips_mode not in ['ips', 'ciips', 'snips', 'dr', 'mrdr', ""]:
            raise ValueError("ips_mode not supported: {}".format(ips_mode))

        self.alpha = alpha
        self.beta = beta
        self.offset = offset
        self.ctr_weight = ctr_weight
        self.dt_weight = dt_weight
        self.loss1, self.loss2, self.loss3 = 0, 0, 0

    def forward(self, input, target, is_calc_inst_weight=None):
        """
        input: list of tensors [input0, input1, input_final]
          - input0: probabilities (after a Sigmoid) for click-class on all samples ([B])
          - input1: logits for dwell-class on clicked samples ([B, C])
          - input_final: logits for the 'united' head ([B, C])
        target: tensor [B, 2], where
          - target[:,0] in {0,1} for click
          - target[:,1] in {0..C-1} for dwell-bucket
        """
        if is_calc_inst_weight is None:
            is_calc_inst_weight = self.is_calc_inst_weight

        input0, input1, input_final = input[0], input[1], input[-1]
        target0, target1 = target[:, 0], target[:, 1]   # binary, regression
        # mask for samples with click==1
        mask = target0 > 0
        pc_time_logits = input1[mask]  # [K, C]
        pc_time_targets = target1[mask]  # [K]
        pc_time_united_logits = input_final[mask]  # [K, C]

        # 1) CTR loss
        self.loss1 = F.binary_cross_entropy(input0, target0.float())  # BCE(mean) on all samples

        # 2) compute inverse propensity score for loss2
        raw_loss2 = F.cross_entropy(pc_time_logits, pc_time_targets, reduction='none')  # CE(none) on clicked samples
        p_click = input0[mask].clone().detach().clamp(min=1e-6)   # [K]
        # print('debug printing p_click---\n', p_click)
        if self.ips_mode == 'ips':
            w = (1.0 / p_click)
            self.loss2 = (raw_loss2 * w).mean()
        elif self.ips_mode == 'ciips':
            # compute clip‐IPS weights on *all* samples, then index by mask
            w = (1.0 / p_click).clamp(max=self.lambda_cap)
            self.loss2 = (raw_loss2 * w).mean()
        elif self.ips_mode == 'snips':
            # ----- SNIPS：self-normalized inverse propensity score -----
            w = 1.0 / p_click  # un-clipped
            self.loss2 = (w * raw_loss2).sum() / w.sum().clamp(min=1e-12)
        elif self.ips_mode == 'dr':
            # ----- Doubly-Robust (DM + IPS residual) -----
            # Direct Method 
            loss_dm = raw_loss2.mean()

            # IPS residual term, probs: [K,C], targets: [K]
            probs = F.softmax(pc_time_logits, dim=1)  # [K,C]
            one_hot = torch.zeros_like(probs) \
                .scatter_(1, pc_time_targets.unsqueeze(1), 1.0)  # [K,C] 真实 one-hot
            residual = one_hot - probs  # (r - r̂)  [K,C]

            p_click = input0[mask].detach().clamp(min=1e-6)  # [K]  观测倾向
            w = 1.0 / p_click  # inverse propensity

            loss_ips = (w.unsqueeze(1) * residual).sum(dim=1).mean()  # ⟨w,(r-r̂)⟩

            self.loss2 = loss_dm + loss_ips  # DR = DM + IPS
        elif self.ips_mode == 'mrdr':
            # Direct Method
            loss_dm = raw_loss2.mean()

            # IPS residual term, probs: [K,C], targets: [K]
            probs = F.softmax(pc_time_logits, dim=1)  # [K,C]
            p_true = probs.gather(1, pc_time_targets.unsqueeze(1)).squeeze(1)  # [K]
            sum_psq = probs.pow(2).sum(dim=1)  # [K]
            res = (1 - p_true).pow(2) + (sum_psq - p_true.pow(2))  # [K]

            w = 1.0 / p_click  # [K]
            loss_ips = (w.unsqueeze(1) * res).sum(dim=1).mean()
            self.loss2 = loss_dm + loss_ips
        else:
            self.loss2 = raw_loss2.mean()

        # 3) compute per-sample weights for united loss
        raw_united = F.cross_entropy(pc_time_united_logits, pc_time_targets, reduction='none')
        if is_calc_inst_weight:
            click_pc = input0[mask].detach()       # [K]
            w1 = F.binary_cross_entropy(
                click_pc,
                target0[mask].float(),
                reduction='none'
            ).pow(self.ctr_weight)
            w2 = F.cross_entropy(
                pc_time_logits.detach(),
                pc_time_targets,
                reduction='none'
            ).pow(self.dt_weight)
            weights = F.relu(w1 * w2 + self.offset)       # [K]
            if weights.sum() > 1e-2:  # normalize weights to sum = K (number of clicked samples)
                weights = (weights / weights.sum() * click_pc.size(0)).detach()
            else:
                weights = torch.ones_like(weights)
            self.loss3 = (raw_united * weights).mean()
        else:
            self.loss3 = raw_united.mean()

        # 4) combined loss
        total_loss = self.loss1 + self.alpha * self.loss2 + self.beta * self.loss3
        return total_loss

