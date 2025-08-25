#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import os
import json
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from numpy.random import choice
import tqdm
import random
from itertools import chain
from collections import defaultdict
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, ndcg_score, log_loss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
try:  # Try to import root_mean_squared_error for sklearn >= 1.4
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    # Fallback for sklearn >= 0.22 and < 1.4: use mean_squared_error(squared=False)
    root_mean_squared_error = lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False)

from preprocess.industry import IndustryDataset
from preprocess.tenrec import TenRec
from model.orca import ORCA, Causal_Weighted_Loss
from model.mbple import MetaBalance


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


def get_model(name, dataset, task_types, config):
    """
    Build the ORCA model
    """
    feature_dims = dataset.feature_dims
    time_dim = len(dataset.time_map)
    tower_dims = config.tower_dims

    if name == 'orca':
        return ORCA(feature_dims=feature_dims, embed_dim=16, num_shared_experts=8,
                   num_specific_experts=8, experts_dims=((128, 64), (32,)), tower_dims=tower_dims,
                   task_types=task_types, dropout=0.2, time_dim=time_dim,
                   base_model=config.base_model, use_inter=config.use_inter)
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
        # result_loss = result_dict['united_loss']
        if result_loss < self.best_loss:
            self.best_loss = result_loss
            self.trial_counter = 0
            if type(optimizer) is dict:
                save_dict = {'epoch_i': epoch_i + 1, 'state_dict': model.state_dict(), 'best_loss': self.best_loss}
                torch.save(save_dict, self.save_path)
            else:
                torch.save({'epoch_i': epoch_i + 1, 'state_dict': model.state_dict(), 'best_loss': self.best_loss,
                            'optimizer': optimizer.state_dict()}, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False



def get_test_result(targets, predicts, user_id, doc_id, time_result_file, criterion_coe, task_types, mode, time_map, ori_time):
    """
    Compute evaluation metrics and (in test mode) dump per-sample predictions.

    Notes:
        - For MTL, dwell-time (DT) metrics are computed on clicked samples only.
        - Writes a CSV with ground-truth and predictions when mode == 'test'.
        - Aggregates both raw DT head and final debiased head metrics.

    Args:
        targets, predicts (dict): Dicts keyed by task_types plus 'final' for ORCA.
        user_id, doc_id (List): Per-sample identifiers.
        time_result_file (str): Output file path (.txt, corresponding .csv is written).
        criterion_coe (List[float]): Loss weights [alpha, beta].
        task_types (List[str]): Tasks (e.g., ['binary','multiclass']).
        mode (str): 'valid' or 'test'.
        time_map (dict): Bucket index -> time mapping.
        ori_time (List[float]): Original read-time values.

    Returns:
        dict: Aggregated metric dictionary.
    """
    result_dict = dict()
    if 'binary' in task_types:  # Single-task CTR uses AUC as the main metric
        result_dict['AUC'] = roc_auc_score(targets['binary'], predicts['binary'])
        result_dict['LogLoss'] = log_loss(targets['binary'], predicts['binary'])

    if len(task_types) > 1:     # For DT, keep only clicked items
        tmp_df = pd.DataFrame({'binary': targets['binary'], 'multi_targets': targets['multiclass'],
                               'binary_predicts': predicts['binary'],
                               'multi_predicts': predicts['multiclass'], 'final_predicts': predicts['final'],
                               'uin': user_id, 'doc_id': doc_id,})
        tmp_df = tmp_df.loc[tmp_df['binary'] > 0]  # Keep only clicked samples
        targets['multiclass'] = tmp_df['multi_targets'].to_list()
        predicts['multiclass'] = tmp_df['multi_predicts'].to_list()
        predicts['final'] = tmp_df['final_predicts'].to_list()
        user_id = tmp_df['uin'].to_list()
        doc_id = tmp_df['doc_id'].to_list()
        clicked_ctr_predicts = tmp_df['binary_predicts'].to_list()
        result_dict.update(
            {'united_loss': log_loss(targets['binary'], predicts['binary']) + criterion_coe[0] * log_loss(
                targets['multiclass'], predicts['multiclass']) + criterion_coe[1] * log_loss(targets['multiclass'],
                                                                                             predicts['final'])})

    if 'multiclass' in task_types:
        result_dict.update({'cross_entropy': log_loss(targets['multiclass'], predicts['multiclass'])})
        result_dict.update({'cross_entropy_final': log_loss(targets['multiclass'], predicts['final'])})
        if mode == 'test':
            # Classification metrics for raw DT head
            predicts_multiclass = np.argmax(predicts['multiclass'], 1)
            result_dict.update(
                {'MSE_class': mean_squared_error(targets['multiclass'], predicts_multiclass),
                 'MAE_class': mean_absolute_error(targets['multiclass'], predicts_multiclass),
                 'RMSE_class': root_mean_squared_error(targets['multiclass'], predicts_multiclass),
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

            # Classification metrics for final (debiased) DT head
            predicts_final = np.argmax(predicts['final'], 1)
            result_dict.update(
                {'MSE_f_class': mean_squared_error(targets['multiclass'], predicts_final),
                 'MAE_f_class': mean_absolute_error(targets['multiclass'], predicts_final),
                 'RMSE_f_class': root_mean_squared_error(targets['multiclass'], predicts_final),
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

            # Regression-style metrics by mapping bucket -> time
            if 'binary' in task_types:
                tmp_df = pd.DataFrame({'binary': targets['binary'], 'ori_time': ori_time})
                targets_t = tmp_df['ori_time'].loc[tmp_df['binary'] > 0].to_list()  # Ground-truth read time
            else:
                targets_t = ori_time
            predicts_t = list(map(time_map.get, np.argmax(predicts['multiclass'], 1)))  # raw head
            predicts_t_final = list(map(time_map.get, np.argmax(predicts['final'], 1)))  # debiased head

            # Per-user NDCG (k=3/5)
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
                                'RMSE': root_mean_squared_error(targets_t, predicts_t),
                                'NDCG@3': np.mean(ncdg3),
                                'NDCG@5': np.mean(ncdg5)
                                })

            result_dict.update({'MSE_f': mean_squared_error(targets_t, predicts_t_final),
                                'MAE_f': mean_absolute_error(targets_t, predicts_t_final),
                                'RMSE_f': root_mean_squared_error(targets_t, predicts_t_final),
                                'NDCG@3_f': np.mean(ncdg3_f),
                                'NDCG@5_f': np.mean(ncdg5_f)
                                })

            # Write per-sample outputs
            print_file = open(time_result_file, mode='w+')
            print(list(result_dict.items()), file=print_file)
            print(f'targets, predicts,  predicts_final  #{len(targets_t)}', file=print_file)
            for i in range(len(targets_t)):
                print(f'{targets_t[i]:8.2f}, {predicts_t[i]:8.2f}, {predicts_t_final[i]:8.2f}', file=print_file)
            print_file.close()

    if mode == 'test':
        # ---------- extra info for each clicked sample ----------
        bucket_pred = np.argmax(predicts['multiclass'], 1)  # predicted bucket (raw)
        bucket_pred_final = np.argmax(predicts['final'], 1)  # predicted bucket (final/debiased)

        # ---------- save to CSV ----------
        save_df = pd.DataFrame({
            'user_id': user_id,
            'doc_id': doc_id,
            'ctr_pred': clicked_ctr_predicts,
            'dt_true_bucket': targets['multiclass'],
            'dt_pred_bucket': bucket_pred,
            'dt_pred_final': bucket_pred_final,
            'dt_true_time': targets_t,
            'dt_pred_time': predicts_t,
            'dt_pred_time_f': predicts_t_final
        })
        save_df.to_csv(time_result_file.replace('.txt', '.csv'), index=False)

    return result_dict


def train(model, optimizer, data_loader, criterion, device, log_interval=20, print_file=None, config=None):
    """
    Train loop for ORCA.

    Supports:
      - Feature masking (config.use_mask_fea / mask_fea_ratio).
      - MetaBalance optimization for shared/task layers when base_model includes 'balance'.
      - Separate optimizer for 'united' (bias) tower when use_mask_fea is enabled.

    Logs averaged losses every `log_interval` steps.
    """
    if config.use_mask_fea:
        optimizer_united = optimizer['united']
        optimizer = optimizer['normal']
    if 'balance' in config.base_model:
        optimizer_meta = optimizer['meta']
        optimizer_taskLayer, optimizer_sharedLayer = optimizer['task'], optimizer['shared']
    model.train()
    total_loss, total_loss1, total_loss2, total_loss3 = 0, 0, 0, 0
    mask_fea_cnt = 0
    tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=10.0)

    for i, batch in enumerate(tk0):
        if len(batch) == 3:  # (x_main , x_mask , y)
            fields, fields_mask, target = batch
            fields = fields.to(device)
            fields_mask = fields_mask.to(device)
        else:  # (x , y)
            fields, target = batch
            fields = fields.to(device)
        target = target.to(device)

        if 'balance' in config.base_model:
            # -- STEP 1: update task‐specific layers normally --
            y = model(fields)
            loss = criterion(y, target)
            model.zero_grad()
            loss.backward()
            optimizer_taskLayer.step()  # only touches task params

            # -- STEP 2: update shared layers with MetaBalance --
            model.zero_grad()
            y = model(fields)  # forward again to recompute loss1, loss2
            loss = criterion(y, target)
            optimizer_meta.step([criterion.loss1, criterion.loss2, criterion.loss3])
            optimizer_sharedLayer.step()
        else:
            y = model(fields)
            loss = criterion(y, target)
            model.zero_grad()
            loss.backward()
            optimizer.step()

        if len(batch) == 3:
            mask_fea_cnt += 1
            y = model(fields_mask)  # forward with masking features
            loss = criterion(y, target)
            model.zero_grad()
            loss.backward()
            optimizer_united.step()
            data_loader.dataset.dataset.change_mode('normal')  # next batch will use normal features

        if config.use_mask_fea and random.random() < config.mask_fea_ratio:
            data_loader.dataset.dataset.change_mode('mask')  # next batch will use mask features

        total_loss += loss.item()
        total_loss1 += criterion.loss1.item()
        total_loss2 += criterion.loss2.item()
        total_loss3 += criterion.loss3.item()
        if (i + 1) % log_interval == 0:
            total_loss /= log_interval
            total_loss1 /= log_interval
            total_loss2 /= log_interval
            total_loss3 /= log_interval
            tk0.set_postfix(total_loss=f"{total_loss:.4f}", loss1=f"{total_loss1:.4f}",
                            loss2=f"{total_loss2:.4f}", loss3=f"{total_loss3:.4f}", mask_fea_cnt=mask_fea_cnt)
            print(f'interval {i+1}:  train loss: {total_loss:.4f}, loss1: {total_loss1:.4f}, '
                  f'loss2:{total_loss2:.4f}, loss3:{total_loss3:.4f}', file=print_file)
            total_loss, total_loss1, total_loss2, total_loss3 = 0, 0, 0, 0
    data_loader.dataset.dataset.change_mode('normal')


def test(model, data_loader, criterion_coe, device, mode='valid', time_map=[], ori_time=[], result_dir=""):
    """
    Evaluation loop.

    - Applies softmax to both raw DT head and final (debiased) head.
    - Collects targets/predictions for CTR and DT.
    - Delegates metric computation and optional CSV dumping to `get_test_result`.
    """
    model.eval()
    task_types = model.task_types
    targets, predicts, user_id, doc_id = defaultdict(list), defaultdict(list), list(), list()
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
            doc_id.extend(fields[:, 1].tolist())  # sparse_fields[1]的column是hashed_docid

    return get_test_result(targets, predicts, user_id, doc_id,
                           f'{result_dir}/dwell time predict result_{model.model_name}.txt',
                           criterion_coe, task_types, mode, time_map, ori_time)


def main_orca(dataset_name, dataset_path, model_name, task_types, epoch, learning_rate,
              batch_size, weight_decay, device, save_dir, result_dir, config):
    """
    Entry point for training/evaluating ORCA.

    Workflow:
      1) Prepare device, seeds, logging files, and dataset/dataloaders.
      2) Build model and set up optimizers (supports MetaBalance and united-head optimizer).
      3) Define causal-weighted loss for ORCA.
      4) Initial test → train/validate with early stopping → load best → final test.
      5) Log metrics (stdout, file, and wandb).

    Args mirror CLI flags passed from main.py.
    """
    device = torch.device(device)
    alpha = config.alpha  # loss weights: loss_ctr + alpha * loss_time + beta*loss_time_f
    beta = config.beta
    setup_random_seed(config.seed)
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
                                                              generator=torch.Generator().manual_seed(config.seed))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    model = get_model(model_name, dataset, task_types, config).to(device)

    if 'balance' in config.base_model:
        optimizer_meta = MetaBalance(model.sharedLayerParameters, relax_factor=config.relax_factor,
                                     beta=config.balance_beta)
        optimizer_sharedLayer = optim.Adam(model.sharedLayerParameters, lr=learning_rate, weight_decay=1e-7)
        optimizer_taskLayer = optim.Adam(model.taskLayerParameters, lr=learning_rate, weight_decay=1e-7)
        optimizer = {'meta': optimizer_meta, 'task': optimizer_taskLayer, 'shared': optimizer_sharedLayer}
    else:
        optimizer = optim.Adam([
            {'params': [p for n, p in model.named_parameters()
                        if n.startswith("united_tower") or n.startswith("united_output_layers")],  # only bias tower
             'lr': learning_rate * config.united_lr_ratio},
            {'params': [p for n, p in model.named_parameters()
                        if not (n.startswith("united_tower") or n.startswith("united_output_layers"))],  # main body
             'lr': learning_rate}
        ], weight_decay=weight_decay)

    # Optimizer to update the united (bias) tower for `Feature-level Counterfactual Intervention`
    if config.use_mask_fea:
        optimizer_united = optim.Adam(
            params=[p for n, p in model.named_parameters()
                    if n.startswith("united_tower") or n.startswith("united_output_layers")],
            lr=learning_rate * config.united_lr_ratio,
            weight_decay=weight_decay)
        optimizer = {'normal': optimizer, 'united': optimizer_united}

    # Causal weighted loss integrates instance weighting and (optional) IPS modes
    criterion = Causal_Weighted_Loss(alpha=alpha, beta=beta, offset=config.offset,
                                     ctr_weight=config.ctr_weight, dt_weight=config.dt_weight,
                                     is_calc_inst_weight=getattr(config, 'use_inst', False),
                                     ips_mode=getattr(config, 'ips_mode', ''),
                                     lambda_cap=getattr(config, 'lambda_cap', 10))

    early_stopper = EarlyStopper(num_trials=config.early_stop, save_path=save_path, task_types=task_types)

    # Initial test before training
    metric_dict = test(model, test_data_loader, [alpha, beta], device, mode='test', time_map=dataset.time_map,
                       ori_time=dataset.data['read_time'].iloc[test_dataset.indices].to_list(), result_dir=result_dir)
    print('init_test:', json.dumps(metric_dict, indent=2, ensure_ascii=False), file=print_file)
    print('init_test:', json.dumps(metric_dict, indent=2, ensure_ascii=False))
    wandb.log(metric_dict)

    # Training loop
    log_interval = 5120 // batch_size if dataset_name != 'tenrec' else 51200 // batch_size
    for epoch_i in range(epoch_s, epoch):
        train(model, optimizer, train_data_loader, criterion, device,
              log_interval=log_interval, print_file=print_file, config=config)

        metric_dict = test(model, valid_data_loader, [alpha, beta], device, mode='valid')  # log result
        print('epoch:', epoch_i, 'validation: ', json.dumps(metric_dict, indent=2, ensure_ascii=False))
        print('epoch:', epoch_i, 'validation: ', json.dumps(metric_dict, indent=2, ensure_ascii=False), file=print_file)
        wandb.log(metric_dict)
        if not early_stopper.is_continuable(model, metric_dict, epoch_i, optimizer):
            print(f'validation: best auc: {early_stopper.best_accuracy:.4f}, best loss: {early_stopper.best_loss:.4f}')
            print(f'validation: best auc: {early_stopper.best_accuracy:.4f}, best loss: {early_stopper.best_loss:.4f}',
                  file=print_file)
            break
            
    # Load best checkpoint and final test
    if os.path.exists(save_path):
        print('loading best model!')
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

    metric_dict = test(model, test_data_loader, [alpha, beta], device, mode='test', time_map=dataset.time_map,
                       ori_time=dataset.data['read_time'].iloc[test_dataset.indices].to_list(), result_dir=result_dir)
    print('test:', json.dumps(metric_dict, indent=2, ensure_ascii=False), file=print_file)
    print('test:', json.dumps(metric_dict, indent=2, ensure_ascii=False))
    wandb.log(metric_dict)

    print_file.close()
