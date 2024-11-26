#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import tqdm
import random
from collections import defaultdict
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, ndcg_score, log_loss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from main_mtc import main_mtc
from preprocess.industry import IndustryDataset
from preprocess.tenrec import TenRec
from model.dfm import DeepFactorizationMachineModel
from model.mmoe import MmoeModel, MultiLoss
from model.ple import ProgressiveLayeredExtraction
from model.afi import AutomaticFeatureInteractionModel
from model.xdfm import ExtremeDeepFactorizationMachineModel
from model.afn import AdaptiveFactorizationNetwork
from model.nfm import NeuralFactorizationMachineModel
from model.dcnv2 import DCNV2
from model.aitm import AitmModel
from model.esmm import ESMM
from model.mbple import MetaBalanceProgressiveLayeredExtraction, MetaBalance


def get_dataset(name, dataset_path, model_name, task_types, print_file):
    if name == 'industry':
        return IndustryDataset(dataset_path, model_name, task_types,
                               ['date1', 'date2', 'date3'], print_file)
    elif name == 'industry_big':
        return IndustryDataset(dataset_path, model_name, task_types,
                               ['date4', 'date5', 'date6', 'date7', 'date8', 'date9', 'date10'], print_file)
    elif name == 'tenrec':
        return TenRec(dataset_path, model_name, task_types, print_file)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, dataset, tower_dims, embed_dim):
    feature_dims = dataset.feature_dims
    task_types = dataset.task_types
    time_dim = len(dataset.time_map)
    if len(task_types) == 1:
        if name == 'nfm':  # one hidden layer
            return NeuralFactorizationMachineModel(feature_dims=feature_dims, embed_dim=embed_dim,
                                                   mlp_dims=(128, 64, 32, 16), task_types=task_types,
                                                   dropouts=(0.2, 0.2), time_dim=time_dim)
        elif name == 'dfm':  # 在这里比wd多了一个二阶交互项
            return DeepFactorizationMachineModel(feature_dims=feature_dims, embed_dim=embed_dim,
                                                 mlp_dims=(128, 64, 32, 16), task_types=task_types,
                                                 dropout=0.2, time_dim=time_dim)
        elif name == 'afi':
            return AutomaticFeatureInteractionModel(feature_dims=feature_dims, embed_dim=embed_dim, atten_embed_dim=64,
                                                    num_heads=2, num_layers=3, mlp_dims=(128, 64, 32, 16),
                                                    task_types=task_types, dropouts=(0.2, 0.2), time_dim=time_dim)
        elif name == 'xdfm':
            return ExtremeDeepFactorizationMachineModel(feature_dims=feature_dims, embed_dim=embed_dim,
                                                        mlp_dims=(128, 64, 32, 16), cross_layer_sizes=(16, 16),
                                                        task_types=task_types, dropout=0.2, time_dim=time_dim)
        elif name == 'afn':
            return AdaptiveFactorizationNetwork(feature_dims=feature_dims, embed_dim=embed_dim, LNN_dim=1500,
                                                lnn_mlp_dims=(400, 400, 400), mlp_dims=(128, 64, 32, 16),
                                                task_types=task_types, dropout=0.2, time_dim=time_dim)
        elif name == 'dcnv2':
            return DCNV2(feature_dims=feature_dims, embed_dim=embed_dim,
                         cross_num=2, dnn_hidden_units=(128, 64, 32), low_rank=32, num_experts=4,
                         task_types=task_types, dropout=0.2, time_dim=time_dim)
        else:
            raise ValueError('unknown single-task model name: ' + name)
    else:
        if name == 'mmoe':
            return MmoeModel(feature_dims=feature_dims, embed_dim=embed_dim, experts_dims=(128, 64, 32), num_experts=8,
                             tower_dims=(16,), task_types=task_types, dropout=0.2, time_dim=time_dim)
        elif name == 'ple':
            return ProgressiveLayeredExtraction(feature_dims=feature_dims, embed_dim=embed_dim, num_shared_experts=8,
                                                num_specific_experts=8, experts_dims=((128, 64), (32,)),
                                                tower_dims=(16,), task_types=task_types, dropout=0.2, time_dim=time_dim)
        elif name == 'aitm':
            return AitmModel(feature_dims=feature_dims, embed_dim=embed_dim, experts_dims=(128, 64, 32),
                             tower_dims=(16,), task_types=task_types, dropout=0.2, time_dim=time_dim)
        elif name == 'esmm':
            return ESMM(feature_dims=feature_dims, embed_dim=embed_dim, tower_dims=(128, 64, 32, 16),
                        task_types=task_types, dropout=0.2, time_dim=time_dim)
        elif name == 'metabalance_ple':  # based on PLE
            return MetaBalanceProgressiveLayeredExtraction(feature_dims=feature_dims, embed_dim=embed_dim,
                                                           num_shared_experts=8, num_specific_experts=8,
                                                           experts_dims=((128, 64), (32,)), tower_dims=(16,),
                                                           task_types=task_types, dropout=0.2, time_dim=time_dim)
        else:
            raise ValueError('unknown multi-task model name: ' + name)


def setup_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class EarlyStopper(object):
    def __init__(self, num_trials, save_path, task_types):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.best_loss = 1e6
        self.save_path = save_path
        self.task_types = task_types

    def is_continuable(self, model, result_dict, epoch_i, optimizer):
        if ('multiclass' in self.task_types) and (len(self.task_types) == 1):
            result_loss = list(result_dict.values())[0]  # 单DT任务取自己的指标进行stop
        if len(self.task_types) > 1:
            result_loss = result_dict['cross_entropy']  # 多任务取时长指标stop
            # result_loss = result_dict['multi_loss']

        if ('binary' in self.task_types) and (len(self.task_types) == 1) and (result_dict['AUC'] > self.best_accuracy):
            self.best_accuracy = result_dict['AUC']
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            torch.save({'epoch': epoch_i + 1, 'state_dict': model.state_dict(), 'best_accuracy': self.best_accuracy,
                        'optimizer': optimizer.state_dict()}, self.save_path)
            return True
        elif ('multiclass' in self.task_types) and (result_loss < self.best_loss):
            self.best_loss = result_loss
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            torch.save({'epoch': epoch_i + 1, 'state_dict': model.state_dict(), 'best_loss': self.best_loss,
                        'optimizer': optimizer.state_dict()}, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def get_test_result(targets, predicts, user_id, time_result_file, alpha, task_types, mode, time_map, ori_time):
    result_dict = dict()
    if 'binary' in task_types:  # 单任务二分类（ctr）时，以auc为评判标准
        result_dict['AUC'] = roc_auc_score(targets['binary'], predicts['binary'])
        result_dict['LogLoss'] = log_loss(targets['binary'], predicts['binary'])

    if len(task_types) > 1:     # dwell time计算只取点击过的物品
        tmp_df = pd.DataFrame({'binary': targets['binary'], 'multi_targets': targets['multiclass'],
                               'multi_predicts': predicts['multiclass'], 'uin': user_id})
        targets['multiclass'] = tmp_df['multi_targets'].loc[tmp_df['binary'] > 0].to_list()
        predicts['multiclass'] = tmp_df['multi_predicts'].loc[tmp_df['binary'] > 0].to_list()
        user_id = tmp_df['uin'].loc[tmp_df['binary'] > 0].to_list()
        result_dict.update({'multi_loss': log_loss(targets['binary'],
                                                   predicts['binary']) + alpha * log_loss(targets['multiclass'],
                                                                                          predicts['multiclass'])})

    if 'multiclass' in task_types:  # 包含多分类任务
        result_dict.update({'cross_entropy': log_loss(targets['multiclass'], predicts['multiclass'])})

        if mode == 'test':  # 如果为test，则返回完整指标
            # 计算多分类相关指标
            predicts_multiclass = np.argmax(predicts['multiclass'], 1)
            result_dict.update(
                {'MSE_class': mean_squared_error(targets['multiclass'], predicts_multiclass),
                 'MAE_class': mean_absolute_error(targets['multiclass'], predicts_multiclass),
                 'RMSE_class': mean_squared_error(targets['multiclass'], predicts_multiclass, squared=False),
                 'Accuracy': accuracy_score(targets['multiclass'], predicts_multiclass),
                 'Macro_Precision': precision_score(targets['multiclass'], predicts_multiclass,
                                                    average='macro', zero_division=0),
                 'Micro_Precision': precision_score(targets['multiclass'], predicts_multiclass,
                                                    average='micro', zero_division=0),
                 'Macro_Recall': recall_score(targets['multiclass'], predicts_multiclass, average='macro'),
                 'Micro_Recall': recall_score(targets['multiclass'], predicts_multiclass, average='micro'),
                 'Macro_F1-score': f1_score(targets['multiclass'], predicts_multiclass, average='macro'),
                 'Micro_F1-score': f1_score(targets['multiclass'], predicts_multiclass, average='micro'),
                 'Weighted_F1-score': f1_score(targets['multiclass'], predicts_multiclass, average='weighted'),
                 'Macro_AUC': roc_auc_score(targets['multiclass'], predicts['multiclass'],
                                            average='macro', multi_class='ovr'),
                 'Weighted_AUC': roc_auc_score(targets['multiclass'], predicts['multiclass'],
                                               average='weighted', multi_class='ovr')
                 })

            # 计算回归相关指标
            if 'binary' in task_types:  # 定义真实的阅读时间：只抽取点击后的阅读时间
                tmp_df = pd.DataFrame({'binary': targets['binary'], 'ori_time': ori_time})
                targets_t = tmp_df['ori_time'].loc[tmp_df['binary'] > 0].to_list()  # 原用户阅读时间
            else:
                targets_t = ori_time
            predicts_t = list(map(time_map.get, np.argmax(predicts['multiclass'], 1)))  # 多分类标签映射回时间
            data_tmp = pd.DataFrame({'user_id': user_id, 'targets': targets_t, 'predicts': predicts_t})
            ncdg3, ncdg5 = list(), list()

            for i in data_tmp['user_id'].unique():
                target_tmp = data_tmp['targets'].loc[data_tmp['user_id'] == i].to_numpy().reshape((1, -1))
                predict_tmp = data_tmp['predicts'].loc[data_tmp['user_id'] == i].to_numpy().reshape((1, -1))

                ncdg3.append(ndcg_score(target_tmp, predict_tmp, k=3) if target_tmp.size != 1 else 1)
                ncdg5.append(ndcg_score(target_tmp, predict_tmp, k=5) if target_tmp.size != 1 else 1)

            result_dict.update({'MSE': mean_squared_error(targets_t, predicts_t),
                                'MAE': mean_absolute_error(targets_t, predicts_t),
                                'RMSE': mean_squared_error(targets_t, predicts_t, squared=False),
                                'NDCG@3': np.mean(ncdg3),
                                'NDCG@5': np.mean(ncdg5)
                                })

            print_file = open(time_result_file, mode='w+')
            print(list(result_dict.items()), file=print_file)
            print(f'targets, predicts  #{len(targets_t)}', file=print_file)
            for i in range(len(targets_t)):
                print(f'{targets_t[i]:8.2f}, {predicts_t[i]:8.2f}', file=print_file)
            print_file.close()

    return result_dict


def train(model, optimizer, data_loader, criterion, device, log_interval=40, print_file=None):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields = fields.to(device)
        target = target.to(device)
        y = model(fields)
        if len(model.task_types) == 1 and ('binary' in model.task_types):
            loss = criterion(y, target.float())
        else:
            loss = criterion(y, target)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            print('interval ', i+1, ' train loss: ', format(total_loss / log_interval, '.4f'))
            print('interval ', i+1, ' train loss: ', format(total_loss / log_interval, '.4f'), file=print_file)
            total_loss = 0


def train_balance(model, optimizer, optimizer_sharedLayer, optimizer_taskLayer, data_loader,
                  criterion, device, log_interval=40, print_file=None):
    model.train()
    total_loss = 0
    for i, (fields, target) in enumerate(data_loader, 0):
        fields = fields.to(device)
        target = target.to(device)
        y = model(fields)
        loss = criterion(y, target)
        model.zero_grad()
        # loss.backward(retain_graph=True)
        loss.backward()
        optimizer_taskLayer.step()

        y = model(fields)
        loss = criterion(y, target)
        model.zero_grad()
        optimizer.step([criterion.loss1, criterion.loss2])
        optimizer_sharedLayer.step()

        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            # tk0.set_postfix(loss=total_loss / log_interval)  # 进度条右边显示信息，‘=’之前自动作为关键字key
            print('interval ', i+1, ' train loss: ', format(total_loss / log_interval, '.4f'))
            print('interval ', i+1, ' train loss: ', format(total_loss / log_interval, '.4f'), file=print_file)
            total_loss = 0


def test(model, data_loader, alpha, device, mode='valid', time_map=[], ori_time=[], result_dir=""):
    model.eval()
    task_types = model.task_types
    targets, predicts, user_id = defaultdict(list), defaultdict(list), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields = fields.to(device)
            target = target.to(device)
            y = model(fields)
            if len(task_types) == 1:
                if 'multiclass' in task_types:
                    y = F.softmax(y, dim=1)
                targets[task_types[0]].extend(target.tolist())
                predicts[task_types[0]].extend(y.tolist())
            else:
                y[1] = F.softmax(y[1], dim=1)
                targets[task_types[0]].extend(target[:, 0].tolist())
                targets[task_types[1]].extend(target[:, 1].tolist())
                predicts[task_types[0]].extend(y[0].tolist())
                predicts[task_types[1]].extend(y[1].tolist())
            user_id.extend(fields[:, 0].tolist())  # sparse_fields[0]的column是uin

    return get_test_result(targets, predicts, user_id,
                           f'{result_dir}/dwell time predict result_{model.model_name}_1.txt',
                           alpha, task_types, mode, time_map, ori_time)


def main(dataset_name, dataset_path, model_name, task_types, epoch, learning_rate,
         batch_size, weight_decay, device, save_dir, result_dir, config):
    device = torch.device(device)
    alpha = config.alpha  # 两个任务的loss比重 loss_ctr + alpha*loss_time
    save_path = save_dir
    print_file = open(f'{result_dir}/{dataset_name}/result_{model_name}_{task_types[0]}.txt', mode='w+')
    result_dir = result_dir+f'/{dataset_name}'
    print(save_path)
    print(config)

    dataset = get_dataset(dataset_name, dataset_path, model_name, task_types, print_file=print_file)

    train_length, valid_length = int(len(dataset) * 0.8), int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = random_split(dataset, (train_length, valid_length, test_length),
                                                              generator=torch.Generator().manual_seed(2022))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    model = get_model(model_name, dataset, config.tower_dims[0], int(config.embed_dim)).to(device)
    if 'balance' in model.model_name:
        optimizer = MetaBalance(model.sharedLayerParameters, relax_factor=config.relax_factor, beta=config.balance_beta)
        optimizer_sharedLayer = optim.Adam(model.sharedLayerParameters, lr=learning_rate, weight_decay=1e-7)
        optimizer_taskLayer = optim.Adam(model.taskLayerParameters, lr=learning_rate, weight_decay=1e-7)
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if len(dataset.task_types) == 1:
        if dataset.task_types[0] == 'binary':
            criterion = torch.nn.BCELoss(reduction='mean')
        else:
            criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = MultiLoss([torch.nn.BCELoss(reduction='mean'), torch.nn.CrossEntropyLoss()], alpha=alpha)
    early_stopper = EarlyStopper(num_trials=config.early_stop, save_path=save_path, task_types=task_types)

    metric_dict = test(model, test_data_loader, alpha, device, mode='test', time_map=dataset.time_map,
                       ori_time=dataset.data['read_time'].iloc[test_dataset.indices].to_list(), result_dir=result_dir)
    print('init_test: ', list(metric_dict.items()))
    print('init_test: ', list(metric_dict.items()), file=print_file)

    for epoch_i in range(epoch):
        if 'balance' in model.model_name:
            train_balance(model, optimizer, optimizer_sharedLayer, optimizer_taskLayer, train_data_loader,
                          criterion, device, log_interval=5120//batch_size, print_file=print_file)
        else:
            train(model, optimizer, train_data_loader, criterion, device,
                  log_interval=5120 // batch_size, print_file=print_file)
        metric_dict = test(model, valid_data_loader, alpha, device, mode='valid')
        print('epoch:', epoch_i, 'validation: ', list(metric_dict.items()))
        print('epoch:', epoch_i, 'validation: ', list(metric_dict.items()), file=print_file)
        if not early_stopper.is_continuable(model, metric_dict, epoch_i, optimizer):
            print(f'validation: best auc: {early_stopper.best_accuracy:.4f}, best loss: {early_stopper.best_loss:.4f}')
            print(f'validation: best auc: {early_stopper.best_accuracy:.4f}, best loss: {early_stopper.best_loss:.4f}',
                  file=print_file)
            break

    if os.path.exists(save_path):
        print('loading best model!')
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    metric_dict = test(model, test_data_loader, alpha, device, mode='test', time_map=dataset.time_map,
                       ori_time=dataset.data['read_time'].iloc[test_dataset.indices].to_list(), result_dir=result_dir)
    print('test: ', list(metric_dict.items()))
    print('test: ', list(metric_dict.items()), file=print_file)

    print_file.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='industry_big')
    parser.add_argument('--dataset_path', default='data/industry.data')
    parser.add_argument('--model_name', default='ple')
    parser.add_argument('--task_num', default=[0, 1])
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--result_dir', default='result')
    parser.add_argument('--tower_dim', default=[0, 0])
    parser.add_argument('--base_model', default='ple')
    parser.add_argument('--united_lr_ratio', type=float, default=1.0)
    parser.add_argument('--united_lr_ratio2', type=float, default=0.5)
    parser.add_argument('--early_stop', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--guide_dim', type=int, default=8)
    parser.add_argument('--alpha', help='原DT任务的权重', default=1)
    parser.add_argument('--beta', help='DT任务误差的权重', default=1)
    parser.add_argument('--tmp', help='用作循环测试的参数', default=0)
    parser.add_argument('--offset', help='mtc-inst模型中weight loss偏移系数', default=-0.2)
    parser.add_argument('--ctr_weight', help='inst模型中weight loss中CTR的加权系数', default=-0.1)
    parser.add_argument('--dt_weight', help='inst模型中weight loss中DT的加权系数', default=0.8)
    parser.add_argument('--inst_interval', help='三种模型叠加时切换优化器的时间', default=200)
    parser.add_argument('--mask_fea_prob', help='在fea方法中MASK特征的概率', default=0.5)
    parser.add_argument('--relax_factor', help='Meta_balance方法中的松弛系数', default=0.7)
    parser.add_argument('--balance_beta', help='Meta_balance方法中的beta', default=0.9)
    args = parser.parse_args()
    config = args
    tmp_list = [(16,), (16, 16), (32, 16), (32,), (32, 32), (16, 8), (16, 16, 16)]
    tower_dims = list()
    for i in config.tower_dim:
        tower_dims.append(tmp_list[i])
    config.tower_dims = tower_dims
    if config.task_num == 0:
        config.task_types = ['binary']
    elif config.task_num == 1:
        config.task_types = ['multiclass']
    else:
        config.task_types = ['binary', 'multiclass']
    config.save_dir = f'save_model/{args.dataset_name}/{args.model_name}_{config.task_types[0]}_{random.randint(1, 500)}.pth.tar'

    if 'mtc' in args.model_name:
        main_mtc(args.dataset_name,
                  args.dataset_path,
                  args.model_name,
                  config.task_types,
                  args.epoch,
                  args.learning_rate,
                  args.batch_size,
                  args.weight_decay,
                  args.device,
                  config.save_dir,
                  args.result_dir,
                  config)
    else:
        main(args.dataset_name,
             args.dataset_path,
             args.model_name,
             config.task_types,
             args.epoch,
             args.learning_rate,
             args.batch_size,
             args.weight_decay,
             args.device,
             config.save_dir,
             args.result_dir,
             config)
