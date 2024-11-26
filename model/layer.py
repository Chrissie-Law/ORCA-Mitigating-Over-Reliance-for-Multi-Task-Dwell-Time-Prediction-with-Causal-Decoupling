#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reference:
    https://github.com/rixwew/pytorch-fm
    https://github.com/shenweichen/DeepCTR-Torch
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class FeaturesLinear(torch.nn.Module):
    # 线性项+偏置项
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)  # 相当于论文中的W
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        # embedding就是论文里的V
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)  # Embedding 也是继承自module，有可学习参数
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)  # 累加除了itemID的各个field长度
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)   # 索引的偏移量
        return self.embedding(x)


class FieldAwareFactorizationMachine(torch.nn.Module):

    def __init__(self, field_dims, field_idx, embed_dim):
        super().__init__()
        self.num_fields = field_idx[-1]  # len(field_dims)
        self.num_features = len(field_dims)
        self.field_idx = field_idx
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)  # 每一个field有自己的embedding
        ])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]  # size (num_fields, (batch_size, num_fields, embed_dim) )
        ix = list()  # [(batch_size, embed_dim)]
        for i in range(self.num_features - 1):
            for j in range(i + 1, self.num_features):
                ix.append(xs[self.field_idx[j]][:, i] * xs[self.field_idx[i]][:, j])  # w_(field_j, i} * w_{field_i, j}
        ix = torch.stack(ix, dim=1)  # 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状
        return ix  # size (batch_size, num_fields*(num_fields-1)/2, embed_dim)


class FactorizationMachine(torch.nn.Module):
    # 求的是二阶特征交互的项
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:  # 在embed_dim方向上求和
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, layer_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for layer_dim in layer_dims:
            layers.append(torch.nn.Linear(input_dim, layer_dim))
            layers.append(torch.nn.BatchNorm1d(layer_dim))
            layers.append(torch.nn.ReLU())  # 原文是tanh，这里改成ReLU
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = layer_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))  # 少了一个activation function，在最终输出的时候加上
        self.mlp = torch.nn.Sequential(*layers)  # list带星号，解开成独立的参数，传入函数；dict同理，key需要与形参名字相同

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


class Expert(torch.nn.Module):
    def __init__(self, input_dim, layer_dims, dropout):
        super(Expert, self).__init__()
        layers = list()
        for layer_dim in layer_dims[:-1]:
            layers.append(torch.nn.Linear(input_dim, layer_dim))
            layers.append(torch.nn.BatchNorm1d(layer_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = layer_dim
        layers.append(torch.nn.Linear(input_dim, layer_dims[-1]))
        self.expert = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim * sparse_feature_num + dense_feature_num)``
        """
        return self.expert(x)


class Tower(torch.nn.Module):
    def __init__(self, input_dim, layer_dims, dropout):
        super(Expert, self).__init__()
        layers = list()
        for layer_dim in layer_dims[:-1]:
            layers.append(torch.nn.Linear(input_dim, layer_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = layer_dim
        layers.append(torch.nn.Linear(input_dim, layer_dims[-1]))
        self.tower = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.tower(x)


class InnerProductNetwork(torch.nn.Module):
    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        return torch.sum(x[:, row] * x[:, col], dim=2)  # 结果应该跟FactorizationMachine一样


class OuterProductNetwork(torch.nn.Module):
    def __init__(self, num_fields, embed_dim, kernel_type='mat'):
        super().__init__()
        num_ix = num_fields * (num_fields - 1) // 2
        if kernel_type == 'mat':  # opnn里论文是outer product，这里实际改成了kernel product
            kernel_shape = embed_dim, num_ix, embed_dim
        elif kernel_type == 'vec':
            kernel_shape = num_ix, embed_dim
        elif kernel_type == 'num':
            kernel_shape = num_ix, 1
        else:
            raise ValueError('unknown kernel type: ' + kernel_type)
        self.kernel_type = kernel_type
        self.kernel = torch.nn.Parameter(torch.zeros(kernel_shape))
        torch.nn.init.xavier_uniform_(self.kernel.data)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]  # size (batch_size, num_ix, embed_dim)
        if self.kernel_type == 'mat':  # (batch_size, 1, num_ix, embed_dim) * (embed_dim, num_ix, embed_dim) = (batch_size, embed_dim, num_ix, embed_dim)
            kp = torch.sum(p.unsqueeze(1) * self.kernel, dim=-1).permute(0, 2, 1)  # permute, 将tensor的维度换位，相当于转置
            # kp.shape = (batch_size, num_ix, embed_dim)
            return torch.sum(kp * q, -1)  # 二阶交互后的矩阵求和后作为标量返回
        else:
            return torch.sum(p * q * self.kernel.unsqueeze(0), -1)


class CrossNetwork(torch.nn.Module):
    def __init__(self, input_dim, num_layers):  # init里是网络里会更新的参数
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)  # 不用ParameterList大概是直接乘不方便
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)  # x^T * w 其实就是各加权后求和
            x = x0 * xw + self.b[i] + x
        return x


class AttentionalFactorizationMachine(torch.nn.Module):
    def __init__(self, embed_dim, attn_size, dropouts):
        super().__init__()
        self.attention = torch.nn.Linear(embed_dim, attn_size)
        self.projection = torch.nn.Linear(attn_size, 1)  # 即2层网络，embed_dim - attn_size - 1
        self.fc = torch.nn.Linear(embed_dim, 1)
        self.dropouts = dropouts

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]  # p,q是二阶特征交互的组合
        inner_product = p * q        # Pair-wise Interaction Layer
        attn_scores = F.relu(self.attention(inner_product))
        attn_scores = F.softmax(self.projection(attn_scores), dim=1)  # 对所有权重归一化
        attn_scores = F.dropout(attn_scores, p=self.dropouts[0], training=self.training)  # 用functional.dropout的时候一定要设training等于模型的training
        attn_output = torch.sum(attn_scores * inner_product, dim=1)   # attention-based pooling
        attn_output = F.dropout(attn_output, p=self.dropouts[1], training=self.training)
        return self.fc(attn_output)


class CompressedInteractionNetwork(torch.nn.Module):
    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cross_layer_sizes[i]
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                                    stride=1, dilation=1, bias=True))
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2  # 对半压缩
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc_input_dim = fc_input_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)  # 通过unsequuze实现外积
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)  # embed_dim始终没变，因此是vector-wise
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        # return self.fc(torch.sum(torch.cat(xs, dim=1), 2))
        return torch.sum(torch.cat(xs, dim=1), 2)


class AnovaKernel(torch.nn.Module):
    def __init__(self, order, reduce_sum=True):
        super().__init__()
        self.order = order
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        batch_size, num_fields, embed_dim = x.shape
        a_prev = torch.ones((batch_size, num_fields + 1, embed_dim), dtype=torch.float).to(x.device)  # DP table的第一行
        for t in range(self.order):
            a = torch.zeros((batch_size, num_fields + 1, embed_dim), dtype=torch.float).to(x.device)
            a[:, t+1:, :] += x[:, t:, :] * a_prev[:, t:-1, :]  # t+1是因为只更新j>t的部分，这里=应该也可以
            a = torch.cumsum(a, dim=1)  # 对应原文的累加
            a_prev = a
        if self.reduce_sum:
            return torch.sum(a[:, -1, :], dim=-1, keepdim=True)
        else:
            return a[:, -1, :]


class AutomaticFeatureInteraction(nn.Module):
    def __init__(self, feature_dims, embed_dim, atten_embed_dim=64, num_heads=2, num_layers=3,
                 dropout=0.2, has_residual=True, output_dim=1,):
        """
        A pytorch implementation of AutoInt.

        Reference:
            W Song, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks, 2018.
        """
        super().__init__()
        self.atten_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)
        self.embed_output_dim = len(feature_dims) * embed_dim
        self.atten_output_dim = len(feature_dims) * atten_embed_dim
        self.has_residual = has_residual
        self.self_attns = torch.nn.ModuleList([nn.MultiheadAttention(atten_embed_dim, num_heads, dropout=dropout)
                                               for _ in range(num_layers)])
        self.attn_fc = torch.nn.Linear(self.atten_output_dim, output_dim)
        if self.has_residual:
            self.V_res_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)

    def forward(self, embed_x):
        atten_x = self.atten_embedding(embed_x)  # size: batch_size, num_fields, atten_embed_dim
        cross_term = atten_x.transpose(0, 1)  # size: num_fields, batch_size, atten_embed_dim
        for self_attn in self.self_attns:
            # input size of MultiheadAttention: L(sequence length), N(batch size), E(embedding dimension)
            # input: query, key, value. output: attn_output(same size of input), attn_output_weights(L*L)
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
        cross_term = cross_term.transpose(0, 1)  # batch_size, num_fields, atten_embed_dim
        if self.has_residual:
            V_res = self.V_res_embedding(embed_x)
            cross_term += V_res
        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)  # transpose-contiguous-view
        return self.attn_fc(cross_term)