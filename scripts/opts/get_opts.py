import argparse
import os
import torch
import models
import data
import json


# MISA中的函数
def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Options():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--name', type=str,
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--log_dir', type=str, default='./logs', help='logs are saved here')
        parser.add_argument('--shared_dir', type=str, default='./shared', help='shared are saved here')
        parser.add_argument('--cuda_benchmark', action='store_true', help='use torch cudnn benchmark')
        parser.add_argument('--has_test', action='store_true',
                            help='whether have test. for 10 fold there is test setting, but in 5 fold there is no test')

        # 🌟 核心修复：在此处直接注册 CIF_MMIN 模型及训练脚本所需的特有参数
        # 这样可以防止在 parse 阶段出现 "unrecognized arguments" 错误
        parser.add_argument('--input_dim_a', type=int, default=74)
        parser.add_argument('--input_dim_l', type=int, default=768)
        parser.add_argument('--input_dim_v', type=int, default=47)
        parser.add_argument('--embd_size_a', default=128, type=int)
        parser.add_argument('--embd_size_l', default=128, type=int)
        parser.add_argument('--embd_size_v', default=128, type=int)
        parser.add_argument('--embd_method_a', default='maxpool', type=str)
        parser.add_argument('--embd_method_v', default='maxpool', type=str)
        parser.add_argument('--AE_layers', type=str, default='256,128,64')
        parser.add_argument('--n_blocks', type=int, default=5)
        parser.add_argument('--cls_layers', type=str, default='128,64')
        parser.add_argument('--dropout_rate', type=float, default=0.5)
        parser.add_argument('--pretrained_path', type=str, default='', help='path to teacher model')
        parser.add_argument('--ce_weight', type=float, default=1.0)
        parser.add_argument('--mse_weight', type=float, default=8.0)
        parser.add_argument('--consistent_weight', type=float, default=100.0)
        parser.add_argument('--kl_weight', type=float, default=0.05)
        parser.add_argument('--mamba_d_state', type=int, default=16)
        parser.add_argument('--corpus_name', type=str, default='MOSI')
        parser.add_argument('--output_dim', type=int, default=1)
        parser.add_argument('--cvNo', type=int, default=1)

        # model parameters
        parser.add_argument('--model', type=str, default='mmin',
                            help='chooses which model to use. [autoencoder | siamese | emotion_A]')
        parser.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay when training')
        parser.add_argument('--init_type', type=str, default='kaiming',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.012,
                            help='scaling factor for normal, xavier and orthogonal.')

        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='multimodal',
                            help='chooses how datasets are loaded. [iemocap, ami, mix]')
        parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset.')

        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix')

        ## training parameter
        parser.add_argument('--print_freq', type=int, default=100,
                            help='frequency of showing training results on console')
        parser.add_argument('--save_epoch_freq', type=int, default=1,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='starting epoch count')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--niter', type=int, default=20, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=80,
                            help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')

        # MISA的参数 (保留)
        parser.add_argument('--mode', type=str, default='train')
        parser.add_argument('--runs', type=int, default=5)
        parser.add_argument('--use_bert', type=str2bool, default=True)
        parser.add_argument('--use_cmd_sim', type=str2bool, default=True)
        parser.add_argument('--eval_batch_size', type=int, default=10)
        parser.add_argument('--n_epoch', type=int, default=500)
        parser.add_argument('--patience', type=int, default=6)
        parser.add_argument('--diff_weight', type=float, default=0.15)
        parser.add_argument('--sim_weight', type=float, default=0.025)
        parser.add_argument('--sp_weight', type=float, default=0.0)
        parser.add_argument('--recon_weight', type=float, default=0.025)
        parser.add_argument('--cls_weight', type=float, default=1)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--optimizer', type=str, default='Adam')
        parser.add_argument('--clip', type=float, default=1.0)
        parser.add_argument('--rnncell', type=str, default='lstm')
        parser.add_argument('--embedding_size', type=int, default=300)
        parser.add_argument('--hidden_size', type=int, default=128)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--reverse_grad_weight', type=float, default=1.0)
        parser.add_argument('--activation', type=str, default='relu')
        parser.add_argument('--random_seed', default=336, type=int, help='# the random seed')
        parser.add_argument('--run_idx', type=str, help='experiment number; for repeat experiment')

        self.isTrain = True
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once)."""
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        else:
            # 如果已经初始化过，我们需要确保 self.parser 存在
            if hasattr(self, 'parser'):
                parser = self.parser
            else:
                parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
                parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        try:
            model_option_setter = models.get_option_setter(model_name)
            parser = model_option_setter(parser, self.isTrain)
            opt, _ = parser.parse_known_args()
        except (ImportError, AttributeError):
            # 如果 models.get_option_setter 失败，至少我们已经在 initialize 中注册了核心参数
            pass

        # modify dataset-related parser options
        try:
            dataset_name = opt.dataset_mode
            dataset_option_setter = data.get_option_setter(dataset_name)
            parser = dataset_option_setter(parser, self.isTrain)
        except (ImportError, AttributeError):
            pass

        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options"""
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'

        if opt.verbose:
            print(message)

        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        log_dir = os.path.join(opt.log_dir, opt.name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def save_json(self, opt):
        dictionary = {}
        for k, v in sorted(vars(opt).items()):
            dictionary[k] = v

        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        save_path = os.path.join(expr_dir, '{}_opt.conf'.format(opt.phase))
        json.dump(dictionary, open(save_path, 'w'), indent=4)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain

        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix
            print("Expr Name:", opt.name)

        self.print_options(opt)

        str_ids = str(opt.gpu_ids).split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            try:
                id = int(str_id)
                if id >= 0:
                    opt.gpu_ids.append(id)
            except ValueError:
                continue
        
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        if opt.isTrain:
            self.save_json(opt)

        self.opt = opt
        return self.opt