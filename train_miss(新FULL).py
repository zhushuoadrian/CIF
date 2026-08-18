import os
import time
import numpy as np
import csv  # 🌟 新增：用于将结果写入表格
from opts.get_opts import Options
from data import create_dataset, create_dataset_with_args
from models import create_model
from utils.logger import get_logger, ResultRecorder, LossRecorder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
import random
import pickle

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# =========================================================================
# 🌟 全局统一：计算对标顶会论文的全维度多模态情感分析指标 (Acc-7, Acc-5 等)
# =========================================================================
def calc_comprehensive_metrics(y_true, y_pred):
    test_preds = np.array(y_pred).squeeze()
    test_truth = np.array(y_true).squeeze()

    mae = np.mean(np.absolute(test_preds - test_truth))
    corr = np.corrcoef(test_preds, test_truth)[0][1]

    pred_7 = np.clip(np.round(test_preds), -3, 3)
    truth_7 = np.clip(np.round(test_truth), -3, 3)
    acc7 = accuracy_score(truth_7, pred_7)

    def to_5_class(arr):
        arr_5 = np.zeros_like(arr)
        arr_5[arr <= -2] = -2
        arr_5[(arr > -2) & (arr <= -0.5)] = -1
        arr_5[(arr > -0.5) & (arr < 0.5)] = 0
        arr_5[(arr >= 0.5) & (arr < 2)] = 1
        arr_5[arr >= 2] = 2
        return arr_5
        
    pred_5 = to_5_class(test_preds)
    truth_5 = to_5_class(test_truth)
    acc5 = accuracy_score(truth_5, pred_5)

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
    if len(non_zeros) > 0:
        pred_2_pos_neg = (test_preds[non_zeros] > 0)
        truth_2_pos_neg = (test_truth[non_zeros] > 0)
        acc2_pos_neg = accuracy_score(truth_2_pos_neg, pred_2_pos_neg)
        f1_pos_neg = f1_score(truth_2_pos_neg, pred_2_pos_neg, average='weighted')
    else:
        acc2_pos_neg, f1_pos_neg = 0, 0

    pred_2_non_neg = (test_preds >= 0)
    truth_2_non_neg = (test_truth >= 0)
    acc2_non_neg = accuracy_score(truth_2_non_neg, pred_2_non_neg)
    f1_non_neg = f1_score(truth_2_non_neg, pred_2_non_neg, average='weighted')

    return {
        'MAE': mae,
        'Corr': corr,
        'Acc-7': acc7,
        'Acc-5': acc5,
        'Acc-2_pos_neg': acc2_pos_neg,
        'F1_pos_neg': f1_pos_neg,
        'Acc-2_non_neg': acc2_non_neg,
        'F1_non_neg': f1_non_neg
    }

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def eval(model, val_iter, is_save=False, phase='test', epoch=-1, mode=None):
    model.eval()

    total_pred = []
    total_label = []
    total_miss_type = []
    total_data = 0

    for i, data in enumerate(val_iter):  
        total_data += 1
        model.set_input(data)  
        model.test()
        
        # 🌟 重点修改处：加入 'MOSEI'
        if model.opt.corpus_name not in ['MOSI', 'SIMS', 'MOSEI']:
            pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        else:
            pred = model.pred.detach().cpu().numpy()
            
        label = data['label']
        total_pred.append(pred)
        total_label.append(label)

        if 'miss_type' in data:
            miss_type = np.array(data['miss_type'])
            total_miss_type.append(miss_type)

    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    
    if len(total_miss_type) > 0:
        total_miss_type = np.concatenate(total_miss_type)

    # 🌟 重点修改处：加入 'MOSEI'
    if model.opt.corpus_name not in ['MOSI', 'SIMS', 'MOSEI']:
        acc = accuracy_score(total_label, total_pred)
        uar = recall_score(total_label, total_pred, average='macro')
        f1 = f1_score(total_label, total_pred, average='macro')

        if is_save:
            save_dir = model.save_dir
            np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
            np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

            if len(total_miss_type) > 0:
                for part_name in ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']:
                    part_index = np.where(total_miss_type == part_name)
                    part_pred = total_pred[part_index]
                    part_label = total_label[part_index]
                    if len(part_label) > 0:
                        acc_part = accuracy_score(part_label, part_pred)
                        uar_part = recall_score(part_label, part_pred, average='macro')
                        f1_part = f1_score(part_label, part_pred, average='macro')
                        np.save(os.path.join(save_dir, '{}_{}_pred.npy'.format(phase, part_name)), part_pred)
                        np.save(os.path.join(save_dir, '{}_{}_label.npy'.format(phase, part_name)), part_label)
                        if phase == 'test':
                            recorder_lookup[part_name].write_result_to_tsv({
                                'acc': acc_part,
                                'uar': uar_part,
                                'f1': f1_part
                            }, cvNo=model.opt.cvNo)

        model.train()
        return acc, uar, f1 

    else:
        # 🌟 使用全维度指标 (回归任务: MOSI, SIMS, MOSEI)
        metrics = calc_comprehensive_metrics(total_label, total_pred)

        if is_save:
            save_dir = model.save_dir
            np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
            np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)
            
            if len(total_miss_type) > 0:
                for part_name in ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']:
                    part_index = np.where(total_miss_type == part_name)
                    part_pred = total_pred[part_index]
                    part_label = total_label[part_index]
                    if len(part_label) > 0:
                        part_metrics = calc_comprehensive_metrics(part_label, part_pred)
                        np.save(os.path.join(save_dir, '{}_{}_pred.npy'.format(phase, part_name)), part_pred)
                        np.save(os.path.join(save_dir, '{}_{}_label.npy'.format(phase, part_name)), part_label)
                        if phase == 'test':
                            recorder_lookup[part_name].write_result_to_tsv({
                                'mae': part_metrics['MAE'],
                                'corr': part_metrics['Corr'],
                                'f1': part_metrics['F1_pos_neg']
                            }, cvNo=model.opt.cvNo)

        model.train()
        return metrics

def clean_chekpoints(expr_name, store_epoch):
    root = os.path.join('checkpoints', expr_name)
    if os.path.exists(root):
        for checkpoint in os.listdir(root):
            if not checkpoint.startswith(str(store_epoch) + '_') and checkpoint.endswith('pth'):
                os.remove(os.path.join(root, checkpoint))

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(random_seed)

def multiclass_acc(preds, truths):
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

if __name__ == '__main__':
    opt = Options().parse()  
    
    logger_path = os.path.join(opt.log_dir, opt.name, str(opt.cvNo))  
    if not os.path.exists(logger_path):  
        os.makedirs(logger_path)

    result_dir = os.path.join(opt.log_dir, opt.name, 'results')
    if not os.path.exists(result_dir):  
        os.makedirs(result_dir)

    total_cv = 10 if opt.corpus_name != 'MSP' else 12
    recorder_lookup = { 
        "total": ResultRecorder(os.path.join(result_dir, 'result_total.tsv'), total_cv=total_cv),
        "azz": ResultRecorder(os.path.join(result_dir, 'result_azz.tsv'), total_cv=total_cv),
        "zvz": ResultRecorder(os.path.join(result_dir, 'result_zvz.tsv'), total_cv=total_cv),
        "zzl": ResultRecorder(os.path.join(result_dir, 'result_zzl.tsv'), total_cv=total_cv),
        "avz": ResultRecorder(os.path.join(result_dir, 'result_avz.tsv'), total_cv=total_cv),
        "azl": ResultRecorder(os.path.join(result_dir, 'result_azl.tsv'), total_cv=total_cv),
        "zvl": ResultRecorder(os.path.join(result_dir, 'result_zvl.tsv'), total_cv=total_cv),
    }
    loss_dir = os.path.join(opt.checkpoints_dir, opt.name, 'loss')
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)
    recorder_loss = LossRecorder(os.path.join(loss_dir, 'result_loss.tsv'), total_cv=total_cv,
                                 total_epoch=opt.niter + opt.niter_decay)

    suffix = '_'.join([opt.model, opt.dataset_mode]) 
    logger = get_logger(logger_path, suffix)  

    if opt.has_test:  
        dataset, val_dataset, tst_dataset = create_dataset_with_args(opt, set_name=['trn', 'val', 'tst'])
    else:
        dataset, val_dataset = create_dataset_with_args(opt, set_name=['trn', 'val'])
        tst_dataset_size = 0
        
    dataset_size = len(dataset)  
    if opt.has_test: tst_dataset_size = len(tst_dataset)
    logger.info('The number of training samples = %d' % dataset_size)
    logger.info('The number of testing samples = %d' % tst_dataset_size)

    model = create_model(opt)  
    model.setup(opt)  

    total_iters = 0  
    best_eval_epoch = -1  
    best_eval_acc, best_eval_uar, best_eval_f1, best_eval_corr, best_eval_mae = 0, 0, 0, 0, 10

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):  
        epoch_start_time = time.time()  
        iter_data_time = time.time()  
        epoch_iter = 0  
        loss_add = True

        for i, data in enumerate(dataset):  
            iter_start_time = time.time()  
            total_iters += 1  
            epoch_iter += opt.batch_size
            model.set_input(data)  
            
            # 🌟 动态加入训练缺失，提高鲁棒性
            train_miss_rate = random.uniform(0.0, 0.5)
            model.apply_random_missing(missing_rate=train_miss_rate)
            
            model.optimize_parameters(epoch)  

            if total_iters % opt.print_freq == 0:  
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                logger.info('Cur epoch {}'.format(epoch) + ' loss ' +
                            ' '.join(map(lambda x: '{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(**losses))
            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:  
            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        logger.info('End of training epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate(logger)  

        # eval
        # 🌟 重点修改处：加入 'MOSEI'
        if model.opt.corpus_name not in ['MOSI', 'SIMS', 'MOSEI']:
            acc, uar, f1 = eval(model, val_dataset)
            logger.info('Val result of epoch %d / %d acc %.4f uar %.4f f1 %.4f' % (
                epoch, opt.niter + opt.niter_decay, acc, uar, f1))
        else:
            metrics = eval(model, val_dataset)
            mae, corr, f1 = metrics['MAE'], metrics['Corr'], metrics['F1_pos_neg']
            logger.info('Val result of epoch %d / %d MAE %.4f Corr %.4f F1 %.4f Acc-7 %.4f Acc-5 %.4f Acc-2 %.4f' % (
                epoch, opt.niter + opt.niter_decay, mae, corr, f1, metrics['Acc-7'], metrics['Acc-5'], metrics['Acc-2_pos_neg']))

        # show test result for debugging
        if opt.has_test and opt.verbose:
            # 🌟 重点修改处：加入 'MOSEI'
            if model.opt.corpus_name not in ['MOSI', 'SIMS', 'MOSEI']:
                acc, uar, f1 = eval(model, tst_dataset)
                logger.info('Tst result of epoch %d / %d acc %.4f uar %.4f f1 %.4f' % (
                epoch, opt.niter + opt.niter_decay, acc, uar, f1))
            else:
                metrics = eval(model, tst_dataset)
                mae, corr, f1 = metrics['MAE'], metrics['Corr'], metrics['F1_pos_neg']
                logger.info('Tst result of epoch %d / %d MAE %.4f Corr %.4f F1 %.4f Acc-7 %.4f Acc-5 %.4f Acc-2 %.4f' % (
                epoch, opt.niter + opt.niter_decay, mae, corr, f1, metrics['Acc-7'], metrics['Acc-5'], metrics['Acc-2_pos_neg']))

        # record epoch with best result
        if opt.corpus_name == 'IEMOCAP':
            if uar > best_eval_uar:
                best_eval_epoch = epoch
                best_eval_uar = uar
                best_eval_acc = acc
                best_eval_f1 = f1
            select_metric = 'uar'
            best_metric = best_eval_uar
        elif opt.corpus_name == 'MSP':
            if f1 > best_eval_f1:
                best_eval_epoch = epoch
                best_eval_uar = uar
                best_eval_acc = acc
                best_eval_f1 = f1
            select_metric = 'f1'
            best_metric = best_eval_f1
        # 🌟 重点修改处：加入 'MOSEI'
        elif opt.corpus_name in ['MOSI', 'SIMS', 'MOSEI']:
            if mae < best_eval_mae:
                best_eval_epoch = epoch
                best_eval_mae = mae
                best_eval_corr = corr
                best_eval_f1 = f1
            select_metric = 'MAE'
            best_metric = best_eval_mae
        else:
            raise ValueError(f'corpus name must be IEMOCAP, CMU-MOSI, SIMS, MOSEI, or MSP, but got {opt.corpus_name}')

    logger.info('Best eval epoch %d found with %s %f' % (best_eval_epoch, select_metric, best_metric))
    
    # test
    if opt.has_test:
        logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
        model.load_networks(best_eval_epoch)
        _ = eval(model, val_dataset, is_save=True, phase='val', epoch=best_eval_epoch)
        
        # 🌟 重点修改处：加入 'MOSEI'
        if model.opt.corpus_name not in ['MOSI', 'SIMS', 'MOSEI']:
            acc, uar, f1 = eval(model, tst_dataset, is_save=True, phase='test', epoch=best_eval_epoch)
            logger.info('Tst result acc %.4f uar %.4f f1 %.4f' % (acc, uar, f1))
            recorder_lookup['total'].write_result_to_tsv({
                'acc': acc, 'uar': uar, 'f1': f1
            }, cvNo=opt.cvNo)
        else:
            metrics = eval(model, tst_dataset, is_save=True, phase='test', epoch=best_eval_epoch)
            mae, corr, f1 = metrics['MAE'], metrics['Corr'], metrics['F1_pos_neg']
            logger.info('Tst result MAE %.4f Corr %.4f F1 %.4f Acc-7 %.4f Acc-5 %.4f Acc-2 %.4f' % (
                mae, corr, f1, metrics['Acc-7'], metrics['Acc-5'], metrics['Acc-2_pos_neg']))
            recorder_lookup['total'].write_result_to_tsv({
                'mae': mae, 'corr': corr, 'f1': f1
            }, cvNo=opt.cvNo)

        # =====================================================================
        # 🌟 顶会级鲁棒性自动化测试大循环 (并自动保存至 CSV)
        # =====================================================================
        logger.info("\n" + "="*50)
        logger.info("开始执行顶会级鲁棒性测试 (随机缺失率 0% -> 90%)")
        logger.info("="*50)
        
        missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        # 🌟 关键：确保模型处于测试状态，关闭内部 Dropout 造成的额外干扰
        model.eval() 
        
        robust_csv_path = os.path.join(result_dir, 'robustness_results.csv')
        file_exists = os.path.exists(robust_csv_path)
        
        with open(robust_csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                # 🌟 重点修改处：加入 'MOSEI'
                if model.opt.corpus_name in ['MOSI', 'SIMS', 'MOSEI']:
                    writer.writerow(['Fold', 'Missing_Rate', 'MAE', 'Corr', 'Acc-7', 'Acc-5', 'Acc-2(-/non-)', 'F1(-/non-)', 'Acc-2(-/+)', 'F1(-/+)'])
                else:
                    writer.writerow(['Fold', 'Missing_Rate', 'Acc', 'UAR', 'F1'])

            for rate in missing_rates:
                all_preds = []
                all_labels = []
                
                for i, data in enumerate(tst_dataset):
                    model.set_input(data)
                    model.apply_random_missing(missing_rate=rate)
                    model.test()
                    
                    # 🌟 重点修改处：加入 'MOSEI'
                    if model.opt.corpus_name not in ['MOSI', 'SIMS', 'MOSEI']:
                        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
                    else:
                        pred = model.pred.detach().cpu().numpy()
                        
                    label = data['label'].numpy()
                    all_preds.append(pred)
                    all_labels.append(label)
                    
                all_preds = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)
                
                logger.info(f"-> 缺失率 Missing Rate: {int(rate*100)}%")
                
                # 🌟 重点修改处：加入 'MOSEI'
                if model.opt.corpus_name in ['MOSI', 'SIMS', 'MOSEI']:
                    metrics = calc_comprehensive_metrics(all_labels, all_preds)
                    logger.info(f"   [回归] MAE: {metrics['MAE']:.4f} | Corr: {metrics['Corr']:.4f}")
                    logger.info(f"   [细粒度] Acc-7: {metrics['Acc-7']:.4f} | Acc-5: {metrics['Acc-5']:.4f}")
                    logger.info(f"   [二分类(-/non-)] Acc-2: {metrics['Acc-2_non_neg']:.4f} | F1: {metrics['F1_non_neg']:.4f}")
                    logger.info(f"   [二分类(-/+)] Acc-2: {metrics['Acc-2_pos_neg']:.4f} | F1: {metrics['F1_pos_neg']:.4f}")
                    
                    writer.writerow([
                        opt.cvNo,
                        f"{int(rate*100)}%",
                        f"{metrics['MAE']:.4f}",
                        f"{metrics['Corr']:.4f}",
                        f"{metrics['Acc-7']:.4f}",
                        f"{metrics['Acc-5']:.4f}",
                        f"{metrics['Acc-2_non_neg']:.4f}",
                        f"{metrics['F1_non_neg']:.4f}",
                        f"{metrics['Acc-2_pos_neg']:.4f}",
                        f"{metrics['F1_pos_neg']:.4f}"
                    ])
                else:
                    acc = accuracy_score(all_labels, all_preds)
                    uar = recall_score(all_labels, all_preds, average='macro')
                    f1_val = f1_score(all_labels, all_preds, average='macro')
                    logger.info(f"   [分类] Acc: {acc:.4f} | UAR: {uar:.4f} | F1: {f1_val:.4f}")
                    writer.writerow([opt.cvNo, f"{int(rate*100)}%", f"{acc:.4f}", f"{uar:.4f}", f"{f1_val:.4f}"])
                    
                logger.info("-" * 50)
        logger.info("================ 鲁棒性测试结束 ================\n")

    else:
        # 🌟 重点修改处：加入 'MOSEI'
        if model.opt.corpus_name not in ['MOSI', 'SIMS', 'MOSEI']:
            recorder_lookup['total'].write_result_to_tsv({
                'acc': best_eval_acc,
                'uar': best_eval_uar,
                'f1': best_eval_f1
            }, cvNo=opt.cvNo)
        else:
            recorder_lookup['total'].write_result_to_tsv({
                'mae': best_eval_mae,
                'corr': best_eval_corr,
                'f1': best_eval_f1
            }, cvNo=opt.cvNo)

    clean_chekpoints(opt.name + '/' + str(opt.cvNo), best_eval_epoch)