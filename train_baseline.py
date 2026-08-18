import os
import time
import numpy as np
from opts.get_opts import Options
from data import create_dataset_with_args
from models import create_model
from utils.logger import get_logger, ResultRecorder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
import torch
from random import random
import pickle

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def eval(model, val_iter, is_save=False, phase='test', eopch=-1, mode=None):
    # print('eval begin')
    model.eval()
    total_pred = []
    total_label = []

    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.test()
        
        # 🌟 修改：支持 MOSEI 回归预测
        if model.opt.corpus_name not in ['MOSI', 'SIMS', 'MOSEI']:
            pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        else:
            pred = model.pred.detach().cpu().numpy()
            
        label = data['label']
        total_pred.append(pred)
        total_label.append(label)

    # calculate metrics
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    
    # 🌟 修改：区分分类任务和回归任务 (MOSI, SIMS, MOSEI)
    if model.opt.corpus_name not in ['MOSI', 'SIMS', 'MOSEI']:
        acc = accuracy_score(total_label, total_pred)
        uar = recall_score(total_label, total_pred, average='macro')
        f1 = f1_score(total_label, total_pred, average='macro')
        cm = confusion_matrix(total_label, total_pred)
        model.train()

        # save test results
        if is_save:
            save_dir = model.save_dir
            np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
            np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)
        return acc, uar, f1, cm
    else:
        accuracy, mae, f_score = calc_metrics(total_label, total_pred, mode)
        model.train()

        # save test results
        if is_save:
            save_dir = model.save_dir
            np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
            np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

        return accuracy, mae, f_score

def clean_chekpoints(expr_name, store_epoch):
    root = os.path.join(opt.checkpoints_dir, expr_name)
    if os.path.exists(root):
        for checkpoint in os.listdir(root):
            if not checkpoint.startswith(str(store_epoch) + '_') and checkpoint.endswith('pth'):
                os.remove(os.path.join(root, checkpoint))

def calc_metrics(y_true, y_pred, mode=None, to_print=False):
    """
    Metric scheme adapted from:
    https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
    """
    test_preds = y_pred.squeeze(1) if y_pred.ndim > 1 else y_pred
    test_truth = y_true

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    # mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]

    if len(non_zeros) > 0:
        f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    else:
        f_score = 0

    # non-neg - neg
    binary_truth = (test_truth >= 0)
    binary_preds = (test_preds >= 0)

    return accuracy_score(binary_truth, binary_preds), corr, f_score

def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

if __name__ == '__main__':
    opt = Options().parse()  # get training options
    logger_path = os.path.join(opt.log_dir, opt.name, str(opt.cvNo))  # get logger path
    if not os.path.exists(logger_path):  # make sure logger path exists
        os.makedirs(logger_path)

    total_cv = 10 if opt.corpus_name != 'MSP' else 12
    result_recorder = ResultRecorder(os.path.join(opt.log_dir, opt.name, 'result.tsv'),
                                     total_cv=total_cv)  # init result recoreder
    suffix = '_'.join([opt.model, opt.dataset_mode])  # get logger suffix
    logger = get_logger(logger_path, suffix)  # get logger
    if opt.has_test:  # create a dataset given opt.dataset_mode and other options
        dataset, val_dataset, tst_dataset = create_dataset_with_args(opt, set_name=['trn', 'val', 'tst'])
    else:
        dataset, val_dataset = create_dataset_with_args(opt, set_name=['trn', 'val'])
        tst_dataset_size = 0

    dataset_size = len(dataset)  # get the number of images in the dataset.
    logger.info('The number of training samples = %d' % dataset_size)
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    total_iters = 0  # the total number of training iterations
    best_eval_acc, best_eval_uar, best_eval_f1, best_eval_mae = 0, 0, 0, 0
    best_eval_epoch = -1  # record the best eval epoch
    best_loss = 100

    for epoch in range(opt.epoch_count,
                       opt.niter + opt.niter_decay + 1):  # outer loop for different epochs
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch
        total_loss = 0

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            total_iters += 1  # opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters(epoch)  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                logger.info('Cur epoch {}'.format(epoch) + ' loss ' +
                            ' '.join(map(lambda x: '{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(**losses))
                for loss in losses.values():
                    total_loss += loss

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        logger.info('End of training epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate(logger)  # update learning rates at the end of every epoch.

        # eval val set
        # 🌟 修改：支持 MOSEI 验证集评估
        if opt.corpus_name not in ['MOSI', 'SIMS', 'MOSEI']:
            acc, uar, f1, cm = eval(model, val_dataset)
            logger.info('Val result of epoch %d / %d acc %.4f uar %.4f f1 %.4f' % (
                epoch, opt.niter + opt.niter_decay, acc, uar, f1))
            logger.info('\n{}'.format(cm))
        else:
            acc, mae, f1 = eval(model, val_dataset)
            logger.info('Val result of epoch %d / %d acc(non-neg) %.4f Corr %.4f f1 %.4f' % (
                epoch, opt.niter + opt.niter_decay, acc, mae, f1)) # 注意 baseline 中 mae 变量其实返回的是 corr

        # show test result for debugging
        if opt.has_test and opt.verbose:
            # 🌟 修改：支持 MOSEI 测试集评估
            if opt.corpus_name not in ['MOSI', 'SIMS', 'MOSEI']:
                acc, uar, f1, cm = eval(model, tst_dataset)
                logger.info('Tst result of epoch %d / %d acc %.4f uar %.4f f1 %.4f' % (
                    epoch, opt.niter + opt.niter_decay, acc, uar, f1))
                logger.info('\n{}'.format(cm))
            else:
                acc, mae, f1 = eval(model, tst_dataset)
                logger.info('Tst result of epoch %d / %d acc(non-neg) %.4f Corr %.4f f1 %.4f' % (
                    epoch, opt.niter + opt.niter_decay, acc, mae, f1))

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
            
        # 🌟 修改：以 F1 为最佳模型选择依据 (MOSI 和 MOSEI 一致)
        elif opt.corpus_name in ['MOSI', 'SIMS', 'MOSEI']:
            if f1 > best_eval_f1:
                best_eval_epoch = epoch
                best_eval_f1 = f1
                best_eval_acc = acc
                best_eval_mae = mae
            select_metric = 'f1'
            best_metric = best_eval_f1
        else:
            raise ValueError(f'corpus name must be IEMOCAP, MSP, CMU-MOSI, SIMS or MOSEI, but got {opt.corpus_name}')

    # print best eval result
    logger.info('Best eval epoch %d found with %s %f' % (best_eval_epoch, select_metric, best_metric))

    # test
    if not opt.has_test:
        logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
        model.load_networks(best_eval_epoch)
        
        # 🌟 修改：支持 MOSEI 最终测试保存
        if opt.corpus_name not in ['MOSI', 'SIMS', 'MOSEI']:
            _ = eval(model, val_dataset, is_save=True, phase='val')
            acc, uar, f1, cm = eval(model, tst_dataset, is_save=True, phase='test', epoch=best_eval_epoch)
            logger.info('Tst result acc %.4f uar %.4f f1 %.4f' % (acc, uar, f1))
            logger.info('\n{}'.format(cm))
            result_recorder.write_result_to_tsv({
                'acc': acc,
                'uar': uar,
                'f1': f1
            }, cvNo=opt.cvNo)
        else:
            _ = eval(model, val_dataset, is_save=True, phase='val')
            acc, mae, f1 = eval(model, tst_dataset, is_save=True, phase='test', epoch=best_eval_epoch)
            logger.info('Tst result acc(non-neg) %.4f Corr %.4f f1 %.4f' % (acc, mae, f1))
            result_recorder.write_result_to_tsv({
                'acc': acc,
                'MAE': mae, # Baseline 中记录了 Corr
                'f1': f1
            }, cvNo=opt.cvNo)

    else:
        # 🌟 修改：处理没有 Test 验证的情况
        if opt.corpus_name not in ['MOSI', 'SIMS', 'MOSEI']:
            result_recorder.write_result_to_tsv({
                'acc': best_eval_acc,
                'uar': best_eval_uar,
                'f1': best_eval_f1
            }, cvNo=opt.cvNo)
        else:
            result_recorder.write_result_to_tsv({
                'acc': best_eval_acc,
                'MAE': best_eval_mae,
                'f1': best_eval_f1
            }, cvNo=opt.cvNo)

    clean_chekpoints(opt.name + '/' + str(opt.cvNo), best_eval_epoch)