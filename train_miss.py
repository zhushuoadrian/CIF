import os
import time
import numpy as np
from opts.get_opts import Options
from data import create_dataset, create_dataset_with_args
from models import create_model
from utils.logger import get_logger, ResultRecorder, LossRecorder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
import random
import pickle

# === [环境修复] 禁止 cuDNN 自动寻找算法，防止 3090/4090 报错 ===
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
# ==========================================================

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def eval(model, val_iter, is_save=False, phase='test', epoch=-1, mode=None):
    model.eval()
    total_pred = []
    total_label = []
    total_miss_type = []
    total_data = 0

    # random_iters = [i for i in range(0, 23)] # 原代码保留

    for i, data in enumerate(val_iter):  # inner loop within one epoch
        total_data += 1
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.test()
        
        # 获取预测结果
        if model.opt.corpus_name != 'MOSI':
            # 分类任务
            pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        else:
            # 回归任务
            pred = model.pred.detach().cpu().numpy()
            
        label = data['label']
        miss_type = np.array(data['miss_type'])

        total_pred.append(pred)
        total_label.append(label)
        total_miss_type.append(miss_type)

    # 拼接结果
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    total_miss_type = np.concatenate(total_miss_type)

    if model.opt.corpus_name != 'MOSI':
        # === 分类指标计算 ===
        acc = accuracy_score(total_label, total_pred)
        uar = recall_score(total_label, total_pred, average='macro')
        f1 = f1_score(total_label, total_pred, average='macro')

        if is_save:
            save_dir = model.save_dir
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
            np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

            # 分类型保存 (part results)
            for part_name in ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']:
                part_index = np.where(total_miss_type == part_name)
                if len(part_index[0]) == 0: continue
                
                part_pred = total_pred[part_index]
                part_label = total_label[part_index]
                
                acc_part = accuracy_score(part_label, part_pred)
                uar_part = recall_score(part_label, part_pred, average='macro')
                f1_part = f1_score(part_label, part_pred, average='macro')
                
                np.save(os.path.join(save_dir, '{}_{}_pred.npy'.format(phase, part_name)), part_pred)
                np.save(os.path.join(save_dir, '{}_{}_label.npy'.format(phase, part_name)), part_label)
                
                if phase == 'test':
                    # 确保 recorder_lookup 在此作用域可用，或者在此处忽略写入
                    # 为防止报错，这里加个 try-except
                    try:
                        recorder_lookup[part_name].write_result_to_tsv({
                            'acc': acc_part, 'uar': uar_part, 'f1': f1_part
                        }, cvNo=model.opt.cvNo)
                    except:
                        pass 

        model.train()
        return acc, uar, f1

    else:
        # === 回归指标计算 ===
        mae, corr, f_score = calc_metrics(total_label, total_pred, mode)

        if is_save:
            save_dir = model.save_dir
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
            np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

            for part_name in ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']:
                part_index = np.where(total_miss_type == part_name)
                if len(part_index[0]) == 0: continue
                
                part_pred = total_pred[part_index]
                part_label = total_label[part_index]
                mae_part, corr_part, f1_part = calc_metrics(part_label, part_pred, mode)
                
                np.save(os.path.join(save_dir, '{}_{}_pred.npy'.format(phase, part_name)), part_pred)
                np.save(os.path.join(save_dir, '{}_{}_label.npy'.format(phase, part_name)), part_label)
                
                if phase == 'test':
                    try:
                        recorder_lookup[part_name].write_result_to_tsv({
                            'mae': mae_part, 'corr': corr_part, 'f1': f1_part
                        }, cvNo=model.opt.cvNo)
                    except:
                        pass

        model.train()
        return mae, corr, f_score

def clean_chekpoints(expr_name, store_epoch):
    root = os.path.join('checkpoints', expr_name)
    if os.path.exists(root):
        for checkpoint in os.listdir(root):
            if not checkpoint.startswith(str(store_epoch) + '_') and checkpoint.endswith('pth'):
                os.remove(os.path.join(root, checkpoint))

def calc_metrics(y_true, y_pred, mode=None, to_print=False):
    test_preds = y_pred.squeeze(1) if y_pred.ndim > 1 else y_pred
    test_truth = y_true
    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

    mae = np.mean(np.absolute(test_preds - test_truth))
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    return mae, corr, f_score

if __name__ == '__main__':
    opt = Options().parse()
    
    # 日志路径处理
    logger_path = os.path.join(opt.log_dir, opt.name, str(opt.cvNo))
    if not os.path.exists(logger_path): os.makedirs(logger_path)

    result_dir = os.path.join(opt.log_dir, opt.name, 'results')
    if not os.path.exists(result_dir): os.makedirs(result_dir)

    total_cv = 10 if opt.corpus_name != 'MSP' else 12
    
    # 结果记录器
    recorder_lookup = {
        "total": ResultRecorder(os.path.join(result_dir, 'result_total.tsv'), total_cv=total_cv),
        "azz": ResultRecorder(os.path.join(result_dir, 'result_azz.tsv'), total_cv=total_cv),
        "zvz": ResultRecorder(os.path.join(result_dir, 'result_zvz.tsv'), total_cv=total_cv),
        "zzl": ResultRecorder(os.path.join(result_dir, 'result_zzl.tsv'), total_cv=total_cv),
        "avz": ResultRecorder(os.path.join(result_dir, 'result_avz.tsv'), total_cv=total_cv),
        "azl": ResultRecorder(os.path.join(result_dir, 'result_azl.tsv'), total_cv=total_cv),
        "zvl": ResultRecorder(os.path.join(result_dir, 'result_zvl.tsv'), total_cv=total_cv),
    }
    
    loss_dir = os.path.join(opt.image_dir, opt.name, 'loss')
    if not os.path.exists(loss_dir): os.makedirs(loss_dir)
    # recorder_loss = LossRecorder(os.path.join(loss_dir, 'result_loss.tsv'), total_cv=total_cv, total_epoch=opt.niter + opt.niter_decay)

    suffix = '_'.join([opt.model, opt.dataset_mode])
    logger = get_logger(logger_path, suffix)

    # 数据集加载
    if opt.has_test:
        dataset, val_dataset, tst_dataset = create_dataset_with_args(opt, set_name=['trn', 'val', 'tst'])
    else:
        dataset, val_dataset = create_dataset_with_args(opt, set_name=['trn', 'val'])
        
    dataset_size = len(dataset)
    if opt.has_test:
        tst_dataset_size = len(tst_dataset)
        logger.info('The number of testing samples = %d' % tst_dataset_size)
    logger.info('The number of training samples = %d' % dataset_size)

    # 模型创建
    model = create_model(opt)
    model.setup(opt)
    
    total_iters = 0
    best_eval_epoch = -1
    best_eval_acc, best_eval_uar, best_eval_f1, best_eval_corr, best_eval_mae = 0, 0, 0, 0, 10

    # 训练主循环
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            total_iters += opt.batch_size # 注意：这里原代码是+=1还是+=batch_size需看具体逻辑，通常 += batch_size
            epoch_iter += opt.batch_size
            
            model.set_input(data)
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

        # 验证 (Validation)
        if model.opt.corpus_name != 'MOSI':
            acc, uar, f1 = eval(model, val_dataset)
            logger.info('Val result of epoch %d / %d acc %.4f uar %.4f f1 %.4f' % (
                epoch, opt.niter + opt.niter_decay, acc, uar, f1))
        else:
            mae, corr, f1 = eval(model, val_dataset)
            logger.info('Val result of epoch %d / %d mae %.4f corr %.4f f1 %.4f' % (
                epoch, opt.niter + opt.niter_decay, mae, corr, f1))

        # 测试 (Test) - 调试用
        if opt.has_test and opt.verbose:
            if model.opt.corpus_name != 'MOSI':
                acc, uar, f1 = eval(model, tst_dataset)
                logger.info('Tst result of epoch %d / %d acc %.4f uar %.4f f1 %.4f' % (
                    epoch, opt.niter + opt.niter_decay, acc, uar, f1))
            else:
                mae, corr, f1 = eval(model, tst_dataset)
                logger.info('Tst result of epoch %d / %d mae %.4f corr %.4f f1 %.4f' % (
                    epoch, opt.niter + opt.niter_decay, mae, corr, f1))

        # 记录最佳 Epoch
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
        elif opt.corpus_name == 'MOSI':
            if mae < best_eval_mae:
                best_eval_epoch = epoch
                best_eval_mae = mae
                best_eval_corr = corr
                best_eval_f1 = f1
            select_metric = 'MAE'
            best_metric = best_eval_mae
        else:
            raise ValueError(f'corpus name must be IEMOCAP, CMU-MOSI, or MSP, but got {opt.corpus_name}')

    logger.info('Best eval epoch %d found with %s %f' % (best_eval_epoch, select_metric, best_metric))

    # 训练结束后：加载最佳模型进行最终测试
    if opt.has_test:
        logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
        model.load_networks(best_eval_epoch)
        # 再跑一遍验证集保存结果
        _ = eval(model, val_dataset, is_save=True, phase='val', epoch=best_eval_epoch)
        
        # 跑测试集
        if model.opt.corpus_name != 'MOSI':
            acc, uar, f1 = eval(model, tst_dataset, is_save=True, phase='test', epoch=best_eval_epoch)
            logger.info('Tst result acc %.4f uar %.4f f1 %.4f' % (acc, uar, f1))
            recorder_lookup['total'].write_result_to_tsv({
                'acc': acc, 'uar': uar, 'f1': f1
            }, cvNo=opt.cvNo)
        else:
            mae, corr, f1 = eval(model, tst_dataset, is_save=True, phase='test', epoch=best_eval_epoch)
            logger.info('Tst result mae %.4f corr %.4f f1 %.4f' % (mae, corr, f1))
            recorder_lookup['total'].write_result_to_tsv({
                'mae': mae, 'corr': corr, 'f1': f1
            }, cvNo=opt.cvNo)

    clean_chekpoints(opt.name + '/' + str(opt.cvNo), best_eval_epoch)