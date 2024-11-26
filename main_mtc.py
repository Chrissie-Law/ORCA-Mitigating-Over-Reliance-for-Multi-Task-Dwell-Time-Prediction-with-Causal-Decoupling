#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import os
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import pandas as pd
import numpy as np
from numpy.random import choice
import tqdm
import random
from itertools import chain
from collections import defaultdict
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, ndcg_score, log_loss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocess.industry import IndustryDataset
from preprocess.tenrec import TenRec
from model.mtc import MTCModel, UnitedLoss
from model.mtcfea import MTCFeaModel
from model.mtcctr import MTCCtrModel, MTCFeaCtrModel
from model.mtcinst import MTCInstModel, MTCFeaInstModel, MTCFeaCtrInstModel, MTCCtrInstModel
from model.mtcinst import UnitedWeightedLossSumCtr, UnitedWeightedLossMultiLoss1Loss2


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


def get_model(name, base_model, dataset, task_types, tower_dims, guide_dim):
    feature_dims = dataset.feature_dims
    time_dim = len(dataset.time_map)
    if name == 'mtc':
        return MTCModel(base_model=base_model, feature_dims=feature_dims, embed_dim=16, num_shared_experts=8,
                        num_specific_experts=8, experts_dims=((128, 64), (32,)),
                        tower_dims=tower_dims, task_types=task_types, dropout=0.2, time_dim=time_dim)
    elif name == 'mtc-fea':
        return MTCFeaModel(base_model=base_model, feature_dims=feature_dims, embed_dim=16, num_shared_experts=8,
                           num_specific_experts=8, experts_dims=((128, 64), (32,)),
                           tower_dims=tower_dims, task_types=task_types, dropout=0.2, time_dim=time_dim)
    elif name == 'mtc-ctr':
        return MTCCtrModel(base_model=base_model, feature_dims=feature_dims, embed_dim=16, num_shared_experts=8,
                           num_specific_experts=8, experts_dims=((128, 64), (32,)), tower_dims=tower_dims,
                           task_types=task_types, dropout=0.2, time_dim=time_dim, guide_dim=0)
    elif name == 'mtc-fea-ctr':
        return MTCFeaCtrModel(base_model=base_model, feature_dims=feature_dims, embed_dim=16, num_shared_experts=8,
                              num_specific_experts=8, experts_dims=((128, 64), (32,)), tower_dims=tower_dims,
                              task_types=task_types, dropout=0.2, time_dim=time_dim, guide_dim=0)
    elif name == 'mtc-inst':
        return MTCInstModel(base_model=base_model, feature_dims=feature_dims, embed_dim=16, num_shared_experts=8,
                            num_specific_experts=8, experts_dims=((128, 64), (32,)),
                            tower_dims=tower_dims, task_types=task_types, dropout=0.2, time_dim=time_dim)
    elif name == 'mtc-fea-inst':
        return MTCFeaInstModel(base_model=base_model, feature_dims=feature_dims, embed_dim=16, num_shared_experts=8,
                               num_specific_experts=8, experts_dims=((128, 64), (32,)), tower_dims=tower_dims,
                               task_types=task_types, dropout=0.2, time_dim=time_dim)
    elif name == 'mtc-ctr-inst':
        return MTCCtrInstModel(base_model=base_model, feature_dims=feature_dims, embed_dim=16, num_shared_experts=8,
                               num_specific_experts=8, experts_dims=((128, 64), (32,)), tower_dims=tower_dims,
                               task_types=task_types, dropout=0.2, time_dim=time_dim, guide_dim=guide_dim)
    elif name == 'mtc-fea-ctr-inst':
        return MTCFeaCtrInstModel(base_model=base_model, feature_dims=feature_dims, embed_dim=16, num_shared_experts=8,
                                  num_specific_experts=8, experts_dims=((128, 64), (32,)), tower_dims=tower_dims,
                                  task_types=task_types, dropout=0.2, time_dim=time_dim, guide_dim=0)
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
        result_loss = result_dict['cross_entropy_final']
        if result_loss < self.best_loss:
            self.best_loss = result_loss
            self.trial_counter = 0
            torch.save({'epoch_i': epoch_i + 1, 'state_dict': model.state_dict(), 'best_loss': self.best_loss,
                        'normal_optimizer': optimizer['normal'].state_dict(),
                        'united_optimizer': optimizer['united'].state_dict()},
                       self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def get_test_result(targets, predicts, user_id, time_result_file, criterion_coe, task_types, mode, time_map, ori_time):
    result_dict = dict()
    if 'binary' in task_types:  # 单任务二分类（ctr）时，以auc为评判标准
        result_dict['AUC'] = roc_auc_score(targets['binary'], predicts['binary'])
        result_dict['LogLoss'] = log_loss(targets['binary'], predicts['binary'])

    if len(task_types) > 1:     # dwell time计算只取点击过的物品
        tmp_df = pd.DataFrame({'binary': targets['binary'], 'multi_targets': targets['multiclass'],
                               'multi_predicts': predicts['multiclass'], 'uin': user_id,
                               'final_predicts':predicts['final']})
        targets['multiclass'] = tmp_df['multi_targets'].loc[tmp_df['binary'] > 0].to_list()
        predicts['multiclass'] = tmp_df['multi_predicts'].loc[tmp_df['binary'] > 0].to_list()
        user_id = tmp_df['uin'].loc[tmp_df['binary'] > 0].to_list()
        predicts['final'] = tmp_df['final_predicts'].loc[tmp_df['binary'] > 0].to_list()
        result_dict.update(
            {'united_loss': log_loss(targets['binary'], predicts['binary']) + criterion_coe[0] * log_loss(
                targets['multiclass'], predicts['multiclass']) + criterion_coe[1] * log_loss(targets['multiclass'],
                                                                                             predicts['final'])})

    if 'multiclass' in task_types:  # 包含多分类任务
        result_dict.update({'cross_entropy': log_loss(targets['multiclass'], predicts['multiclass'])})
        result_dict.update({'cross_entropy_final': log_loss(targets['multiclass'], predicts['final'])})
        if mode == 'test':  # 如果为test，则返回完整指标
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

            predicts_final = np.argmax(predicts['final'], 1)
            result_dict.update(
                {'MSE_f_class': mean_squared_error(targets['multiclass'], predicts_final),
                 'MAE_f_class': mean_absolute_error(targets['multiclass'], predicts_final),
                 'RMSE_f_class': mean_squared_error(targets['multiclass'], predicts_final, squared=False),
                 'f_Accuracy': accuracy_score(targets['multiclass'], predicts_final),
                 'f_Macro_Precision': precision_score(targets['multiclass'], predicts_final,
                                                      average='macro', zero_division=0),
                 'f_Micro_Precision': precision_score(targets['multiclass'], predicts_final,
                                                      average='micro', zero_division=0),
                 'f_Macro_Recall': recall_score(targets['multiclass'], predicts_final, average='macro'),
                 'f_Micro_Recall': recall_score(targets['multiclass'], predicts_final, average='micro'),
                 'f_Macro_F1-score': f1_score(targets['multiclass'], predicts_final, average='macro'),
                 'f_Micro_F1-score': f1_score(targets['multiclass'], predicts_final, average='micro'),
                 'f_Weighted_F1-score': f1_score(targets['multiclass'], predicts_final, average='weighted'),
                 'f_Macro_AUC': roc_auc_score(targets['multiclass'], predicts['final'],
                                              average='macro', multi_class='ovr'),
                 'f_Weighted_AUC': roc_auc_score(targets['multiclass'], predicts['final'],
                                                 average='weighted', multi_class='ovr')
                 })

            if 'binary' in task_types:
                tmp_df = pd.DataFrame({'binary': targets['binary'], 'ori_time': ori_time})
                targets_t = tmp_df['ori_time'].loc[tmp_df['binary'] > 0].to_list()  # 原用户阅读时间
            else:
                targets_t = ori_time
            predicts_t = list(map(time_map.get, np.argmax(predicts['multiclass'], 1)))  # 多分类标签映射回时间
            predicts_t_final = list(map(time_map.get, np.argmax(predicts['final'], 1)))
            data_tmp = pd.DataFrame({'user_id': user_id, 'targets': targets_t,
                                     'predicts': predicts_t, 'predicts_final': predicts_t_final})
            ncdg3, ncdg5 = list(), list()
            ncdg3_f, ncdg5_f = list(), list()

            for i in data_tmp['user_id'].unique():
                target_tmp = data_tmp['targets'].loc[data_tmp['user_id'] == i].to_numpy().reshape((1, -1))
                predict_tmp = data_tmp['predicts'].loc[data_tmp['user_id'] == i].to_numpy().reshape((1, -1))

                ncdg3.append(ndcg_score(target_tmp, predict_tmp, k=3) if target_tmp.size != 1 else 1)
                ncdg5.append(ndcg_score(target_tmp, predict_tmp, k=5) if target_tmp.size != 1 else 1)

                predict_final_tmp = data_tmp['predicts_final'].loc[data_tmp['user_id'] == i].to_numpy().reshape(
                    (1, -1))
                ncdg3_f.append(ndcg_score(target_tmp, predict_final_tmp, k=3) if target_tmp.size != 1 else 1)
                ncdg5_f.append(ndcg_score(target_tmp, predict_final_tmp, k=5) if target_tmp.size != 1 else 1)

            result_dict.update({'MSE': mean_squared_error(targets_t, predicts_t),
                                'MAE': mean_absolute_error(targets_t, predicts_t),
                                'RMSE': mean_squared_error(targets_t, predicts_t, squared=False),
                                'NDCG@3': np.mean(ncdg3),
                                'NDCG@5': np.mean(ncdg5)
                                })

            result_dict.update({'MSE_f': mean_squared_error(targets_t, predicts_t_final),
                                'MAE_f': mean_absolute_error(targets_t, predicts_t_final),
                                'RMSE_f': mean_squared_error(targets_t, predicts_t_final, squared=False),
                                'NDCG@3_f': np.mean(ncdg3_f),
                                'NDCG@5_f': np.mean(ncdg5_f)
                                })

            print_file = open(time_result_file, mode='w+')
            print(list(result_dict.items()), file=print_file)
            print(f'targets, predicts,  predicts_final  #{len(targets_t)}', file=print_file)
            for i in range(len(targets_t)):
                print(f'{targets_t[i]:8.2f}, {predicts_t[i]:8.2f}, {predicts_t_final[i]:8.2f}', file=print_file)
            print_file.close()

    return result_dict


def train(model, optimizer, data_loader, criterion, device, log_interval=20, print_file=None, mode='ple&towerN',
          inst_interval=400, config=None):
    model.train()
    total_loss, total_loss1, total_loss2, total_loss3 = 0, 0, 0, 0
    mask_fea_prob = config.mask_fea_prob
    for i, (fields, target) in enumerate(data_loader, 0):
        fields = fields.to(device)
        target = target.to(device)

        if 'ple' in mode:
            criterion.change_mode('normal')
            model.change_mode('normal')  # 读取未进行block的数据，仅对block fea相关模型有效
            model.united_grad(requires_grad=False)
            y = model(fields)
            loss = criterion(y, target)
            model.zero_grad()
            loss.backward()
            optimizer['normal'].step()  # 优化主模型部分
            # optimizer['united'].step()  # 优化模型Tower C'部分，由于学习率不同，因此需要不同的优化器

        if 'towerN' in mode:
            criterion.change_mode('united')
            model.change_mode('normal')
            model.united_grad(requires_grad=True)  # Tower C'部分可优化
            y = model(fields)
            loss = criterion(y, target)
            model.zero_grad()
            loss.backward()
            optimizer['united'].step()  # 优化模型Tower C'部分，由于学习率不同，因此需要不同的优化器

            if ('fea' in model.model_name) and (choice(np.arange(0, 2), p=[1-mask_fea_prob, mask_fea_prob])):
                # block掉content特征进行训练，只影响united相关
                criterion.change_mode('united')
                model.change_mode('united')  # 读取进行block的数据
                model.united_grad(requires_grad=True)  # Tower C'部分可优化
                y = model(fields)
                loss = criterion(y, target)
                model.zero_grad()
                loss.backward()
                optimizer['united'].step()

        total_loss += loss.item()
        total_loss1 += criterion.loss1.item()
        total_loss2 += criterion.loss2.item()
        total_loss3 += criterion.loss3.item()
        if (i + 1) % log_interval == 0:
            total_loss /= log_interval
            total_loss1 /= log_interval
            total_loss2 /= log_interval
            total_loss3 /= log_interval
            # tk0.set_postfix(loss=total_loss)  # 进度条右边显示信息，‘=’之前自动作为关键字key
            print(f'interval {i+1}:  train loss: {total_loss:.4f}, loss1: {total_loss1:.4f}, '
                  f'loss2:{total_loss2:.4f}, loss3:{total_loss3:.4f}')
            print(f'interval {i+1}:  train loss: {total_loss:.4f}, loss1: {total_loss1:.4f}, '
                  f'loss2:{total_loss2:.4f}, loss3:{total_loss3:.4f}', file=print_file)
            total_loss, total_loss1, total_loss2, total_loss3 = 0, 0, 0, 0

        if (i == inst_interval) and ('inst' in model.model_name) and (('fea' in model.model_name) or ('ctr' in model.model_name)):
            # 加入inst加权训练
            if criterion.name == 'weighted':
                print(f'Check: optimizer is inst-weighted in {inst_interval}')
                continue

            print(f'change optimizer in {inst_interval}')
            for p in model.parameters():
                p.requires_grad = False
            for p in chain(model.united_tower.parameters(), model.united_output_layers.parameters(),
                           model.shared_gates.parameters()):
                p.requires_grad = True
            if 'ctr' in model.model_name:
                for p in model.guide:
                    p.requires_grad = True
            optimizer['united'] = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                                   lr=config.learning_rate * config.united_lr_ratio2,
                                                   weight_decay=config.weight_decay)

            criterion = UnitedWeightedLossMultiLoss1Loss2(['BCELoss', 'CELoss', 'CELoss_none'], alpha=config.alpha,
                                                          beta=config.beta,
                                                          offset=config.offset, ctr_weight=config.ctr_weight)


def test(model, data_loader, criterion_coe, device, mode='valid', time_map=[], ori_time=[], result_dir=""):
    model.eval()
    model.change_mode('normal')
    task_types = model.task_types
    targets, predicts, user_id = defaultdict(list), defaultdict(list), list()
    with torch.no_grad():
        for i, (fields, target) in enumerate(data_loader, 0):
            fields = fields.to(device)
            target = target.to(device)
            y = model(fields)
            y[1] = F.softmax(y[1], dim=1)
            y[-1] = F.softmax(y[-1], dim=1)
            targets[task_types[0]].extend(target[:, 0].tolist())
            targets[task_types[1]].extend(target[:, 1].tolist())
            predicts[task_types[0]].extend(y[0].tolist())
            predicts[task_types[1]].extend(y[1].tolist())
            predicts['final'].extend(y[-1].tolist())
            user_id.extend(fields[:, 0].tolist())  # sparse_fields[0]的column是uin

    return get_test_result(targets, predicts, user_id,
                           f'{result_dir}/dwell time predict result_{model.model_name}.txt',
                           criterion_coe, task_types, mode, time_map, ori_time)


def main_mtc(dataset_name, dataset_path, model_name, task_types, epoch, learning_rate,
              batch_size, weight_decay, device, save_dir, result_dir, config):  # config should be deleted
    device = torch.device(device)
    alpha = config.alpha  # 任务的loss比重 loss_ctr + alpha * loss_time + beta*loss_time_f
    beta = config.beta
    save_path = save_dir  # f'{save_dir}/{dataset_name}/{model_name}.pth.tar'
    print_file = open(f'{result_dir}/{dataset_name}/result_{model_name}.txt', mode='w+')
    result_dir = result_dir+f'/{dataset_name}'
    print(save_path)
    print(config)
    epoch_s = 0

    dataset = get_dataset(dataset_name, dataset_path, model_name, task_types, print_file=print_file)

    train_length, valid_length = int(len(dataset) * 0.8), int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = random_split(dataset, (train_length, valid_length, test_length),
                                                              generator=torch.Generator().manual_seed(2022))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    model = get_model(model_name, config.base_model, dataset, task_types, config.tower_dims, int(config.guide_dim)).to(device)
    optimizer = dict()

    model.united_grad(True)  # 只优化Tower C'部分
    optimizer['united'] = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                           lr=learning_rate * config.united_lr_ratio, weight_decay=weight_decay)
    model.united_grad(False)  # 对模型整体进行优化
    optimizer['normal'] = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                           lr=learning_rate, weight_decay=weight_decay)  # 优化除模型B的部分

    for p in model.parameters():
        p.requires_grad = True

    if ('inst' in model_name) and ('fea' not in model_name) and ('ctr' not in model_name):
        criterion = UnitedWeightedLossMultiLoss1Loss2(['BCELoss', 'CELoss', 'CELoss_none'],
                                                      alpha=alpha, beta=beta, offset=config.offset,
                                                      ctr_weight=config.ctr_weight, dt_weight=config.dt_weight)
    else:
        criterion = UnitedLoss(['BCELoss', 'CELoss', 'CELoss'], alpha=alpha, beta=beta)

    early_stopper = EarlyStopper(num_trials=config.early_stop, save_path=save_path, task_types=task_types)

    metric_dict = test(model, test_data_loader, [alpha, beta], device, mode='test', time_map=dataset.time_map,
                       ori_time=dataset.data['read_time'].iloc[test_dataset.indices].to_list(), result_dir=result_dir)
    print('init_test: ', list(metric_dict.items()))
    print('init_test: ', list(metric_dict.items()), file=print_file)

    for epoch_i in range(epoch_s, epoch):
        if dataset_name != 'tenrec':
            train(model, optimizer, train_data_loader, criterion, device,
                  log_interval=5120//batch_size, print_file=print_file, inst_interval=config.inst_interval,
                  config=config)
        else:
            train(model, optimizer, train_data_loader, criterion, device,
                  log_interval=51200 // batch_size, print_file=print_file, inst_interval=config.inst_interval,
                  config=config)

        # 在第一个epoch训练完成之后可以直接换成sample weight的criterion
        if (epoch_i == 0) and ('inst' in model_name) and (('fea' in model_name) or ('ctr' in model_name)):
            criterion = UnitedWeightedLossMultiLoss1Loss2(['BCELoss', 'CELoss', 'CELoss_none'],
                                                          alpha=alpha, beta=beta, offset=config.offset,
                                                          ctr_weight=config.ctr_weight, dt_weight=config.dt_weight)

        metric_dict = test(model, valid_data_loader, [alpha, beta], device, mode='valid')
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

    metric_dict = test(model, test_data_loader, [alpha, beta], device, mode='test', time_map=dataset.time_map,
                       ori_time=dataset.data['read_time'].iloc[test_dataset.indices].to_list(), result_dir=result_dir)
    print('test: ', list(metric_dict.items()))
    print('test: ', list(metric_dict.items()), file=print_file)

    print_file.close()
