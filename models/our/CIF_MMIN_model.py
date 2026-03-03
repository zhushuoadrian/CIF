import torch
import torch.nn as nn
import os
import json
from collections import OrderedDict
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier, Fusion
from models.networks.autoencoder_2 import ResidualAE
# ⭐ 新增: 导入Mamba Encoder
from models.networks.mamba_encoder import BiMambaEncoder
# 引入融合模块
from models.networks.fusion_opt import OptimizedFusionClassifier
from models.utils.config import OptConfig
from models.utt_self_supervise_model import UttSelfSuperviseModel

# =========================================================================
# [新增] Prompt 容器 (作为低置信度时的补充)
# =========================================================================
class PromptContainer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # 初始化三个模态的提示向量
        self.prompt_A = nn.Parameter(torch.randn(1, opt.embd_size_a))
        self.prompt_V = nn.Parameter(torch.randn(1, opt.embd_size_v))
        self.prompt_L = nn.Parameter(torch.randn(1, opt.embd_size_l))
        
        nn.init.xavier_normal_(self.prompt_A)
        nn.init.xavier_normal_(self.prompt_V)
        nn.init.xavier_normal_(self.prompt_L)

# =========================================================================
# [新增] 置信度门控网络 (Confidence Gate)
# 作用：评估输入特征的质量，输出 0~1 的置信度分数
# =========================================================================
class ConfidenceGate(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid() # 输出归一化到 0~1
        )

    def forward(self, feature, hard_mask=None):
        """
        Args:
            feature: [B, D]
            hard_mask: [B, 1] (可选) 如果提供了硬 Mask (0/1)，可以用来强制截断
        Returns:
            alpha: [B, 1] 置信度
        """
        # 1. 自动评估质量
        alpha = self.gate_net(feature)
        
        # 2. 如果明确知道是缺失的 (hard_mask=0)，则强制置信度为 0
        if hard_mask is not None:
            alpha = alpha * hard_mask
            
        return alpha

class CIFMMINModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='lexical input dim')
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
                            help='audio embedding method,last,mean or atten')
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
                            help='visual embedding method,last,mean or atten')
        parser.add_argument('--AE_layers', type=str, default='128,64,32',
                            help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--n_blocks', type=int, default=3, help='number of AE blocks')
        parser.add_argument('--cls_layers', type=str, default='128,128',
                            help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--pretrained_path', type=str, help='where to load pretrained encoder network')
        parser.add_argument('--pretrained_consistent_path', type=str,
                            help='where to load pretrained consistent encoder network')
        parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of ce loss')
        parser.add_argument('--mse_weight', type=float, default=1.0, help='weight of mse loss')
        parser.add_argument('--cycle_weight', type=float, default=1.0, help='weight of cycle loss')
        parser.add_argument('--consistent_weight', type=float, default=1.0, help='weight of consistent loss')
        parser.add_argument('--share_weight', action='store_true',
                            help='share weight of forward and backward autoencoders')
        parser.add_argument('--image_dir', type=str, default='./consistent_image', help='models image are saved here')
        
        # Mamba 相关参数
        parser.add_argument('--mamba_d_state', type=int, default=16, help='state dimension for mamba')
# ⭐ 新增以下4行
        parser.add_argument('--use_mamba', action='store_true', help='use Mamba encoder instead of LSTM/TextCNN')
        parser.add_argument('--mamba_d_conv', type=int, default=4, help='convolution kernel for mamba')
        parser.add_argument('--mamba_expand', type=int, default=2, help='expansion factor for mamba')
parser.add_argument('--mamba_dropout', type=float, default=0.1, help='dropout for mamba encoder')
        
        # 补全参数
        parser.add_argument('--align_dim', type=int, default=128, help='alignment dimension')
        parser.add_argument('--seq_weight', type=float, default=0.5, help='weight of sequence reconstruction loss')
        
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.loss_names = ['CE', 'mse', 'consistent']
        self.model_names = ['C', 'AE'] 

                # ========================================
        # Modality Encoders: 根据use_mamba参数选择使用Mamba或LSTM
        # ========================================
        self.use_mamba = getattr(opt, 'use_mamba', False)
        
        if self.use_mamba:
            print("==> [CIF-MMIN] Using Mamba-based Modality Encoders")
            
            # ⭐ Audio Mamba Encoder
            self.netA = BiMambaEncoder(
                input_dim=opt.input_dim_a,
                output_dim=opt.embd_size_a,
                d_state=opt.mamba_d_state,
                d_conv=opt.mamba_d_conv,
                expand=opt.mamba_expand,
                dropout=opt.mamba_dropout
            )
            
            # ⭐ Text Mamba Encoder
            self.netL = BiMambaEncoder(
                input_dim=opt.input_dim_l,
                output_dim=opt.embd_size_l,
                d_state=opt.mamba_d_state,
                d_conv=opt.mamba_d_conv,
                expand=opt.mamba_expand,
                dropout=opt.mamba_dropout
            )
            
            # ⭐ Visual Mamba Encoder
            self.netV = BiMambaEncoder(
                input_dim=opt.input_dim_v,
                output_dim=opt.embd_size_v,
                d_state=opt.mamba_d_state,
                d_conv=opt.mamba_d_conv,
                expand=opt.mamba_expand,
                dropout=opt.mamba_dropout
            )
        else:
            print("==> [CIF-MMIN] Using original LSTM/TextCNN Encoders")
            # acoustic model
            self.netA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
            # lexical model
            self.netL = TextCNN(opt.input_dim_l, opt.embd_size_l, dropout=0.5)
            # visual model
            self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
        
        self.model_names.append('A')
        self.model_names.append('L')
        self.model_names.append('V')

        # ========================================
        # Invariance Encoders: 根据use_mamba参数选择使用Mamba或LSTM
        # ========================================
        if self.use_mamba:
            print("==> [CIF-MMIN] Using Mamba-based Invariance Encoders")
            
            # ⭐ Audio Invariance Mamba Encoder
            self.netConA = BiMambaEncoder(
                input_dim=opt.input_dim_a,
                output_dim=opt.embd_size_a,
                d_state=opt.mamba_d_state,
                d_conv=opt.mamba_d_conv,
                expand=opt.mamba_expand,
                dropout=opt.mamba_dropout
            )
            
            # ⭐ Text Invariance Mamba Encoder
            self.netConL = BiMambaEncoder(
                input_dim=opt.input_dim_l,
                output_dim=opt.embd_size_l,
                d_state=opt.mamba_d_state,
                d_conv=opt.mamba_d_conv,
                expand=opt.mamba_expand,
                dropout=opt.mamba_dropout
            )
            
            # ⭐ Visual Invariance Mamba Encoder
            self.netConV = BiMambaEncoder(
                input_dim=opt.input_dim_v,
                output_dim=opt.embd_size_v,
                d_state=opt.mamba_d_state,
                d_conv=opt.mamba_d_conv,
                expand=opt.mamba_expand,
                dropout=opt.mamba_dropout
            )
        else:
            print("==> [CIF-MMIN] Using original LSTM Invariance Encoders")
            self.netConA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
            self.netConL = LSTMEncoder(opt.input_dim_l, opt.embd_size_l)
            self.netConV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
        
        self.model_names.append('ConA')
        self.model_names.append('ConL')
        self.model_names.append('ConV')
        
        # AE model
        AE_layers = list(map(lambda x: int(x), opt.AE_layers.split(',')))
        AE_input_dim = opt.embd_size_a + opt.embd_size_v + opt.embd_size_l
        self.netAE = ResidualAE(AE_layers, opt.n_blocks, AE_input_dim, dropout=0, use_bn=False)
        
        # 分类层
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        cls_input_size = opt.embd_size_a + opt.embd_size_v + opt.embd_size_l
        
        mamba_state = getattr(opt, 'mamba_d_state', 16)
        self.netFusionOpt = OptimizedFusionClassifier(
            input_dim=cls_input_size, 
            output_dim=opt.output_dim,
            dropout=opt.dropout_rate,
            d_state=mamba_state
        )
        self.model_names.append('FusionOpt')

        # =========================================================================
        # [修改点] 软门控核心组件初始化 (注意变量名必须加 net 前缀)
        # =========================================================================
        # 1. 找回 Prompt (作为 alpha 低时的替补)
        self.netPrompts = PromptContainer(opt)
        self.model_names.append('Prompts')

        # 2. 初始化三个模态的门控网络
        # 必须使用 self.netgate_X 命名，以匹配 model_names 中的 'gate_X'
        self.netgate_A = ConfidenceGate(opt.embd_size_a)
        self.netgate_L = ConfidenceGate(opt.embd_size_l)
        self.netgate_V = ConfidenceGate(opt.embd_size_v)
        
        # 将门控加入 model_names
        self.model_names += ['gate_A', 'gate_L', 'gate_V']

        # 兼容旧逻辑的分类器
        if self.opt.corpus_name != 'MOSI':
            self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate,
                                     use_bn=opt.bn)
        else:
            self.netC = Fusion(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate)

        if self.isTrain:
            self.load_pretrained_encoder(opt)
            if self.opt.corpus_name != 'MOSI':
                self.criterion_ce = torch.nn.CrossEntropyLoss()
            else:
                self.criterion_ce = torch.nn.MSELoss()
            self.criterion_mse = torch.nn.MSELoss()
            
            # 优化器
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            
            self.output_dim = opt.output_dim
            self.ce_weight = opt.ce_weight
            self.mse_weight = opt.mse_weight
            self.consistent_weight = opt.consistent_weight
            self.cycle_weight = opt.cycle_weight
        else:
            self.load_pretrained_encoder(opt)

        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        image_save_dir = os.path.join(opt.image_dir, opt.name)
        image_save_dir = os.path.join(image_save_dir, str(opt.cvNo))
        self.predict_image_save_dir = os.path.join(image_save_dir, 'predict')
        self.consistent_image_save_dir = os.path.join(image_save_dir, 'consistent')
        self.loss_image_save_dir = os.path.join(image_save_dir, 'loss')
        if not os.path.exists(self.predict_image_save_dir): os.makedirs(self.predict_image_save_dir)
        if not os.path.exists(self.consistent_image_save_dir): os.makedirs(self.consistent_image_save_dir)
        if not os.path.exists(self.loss_image_save_dir): os.makedirs(self.loss_image_save_dir)

    def load_pretrained_encoder(self, opt):
        print('Init parameter from {}'.format(opt.pretrained_path))
        pretrained_path = os.path.join(opt.pretrained_path, str(opt.cvNo))
        pretrained_config_path = os.path.join(opt.pretrained_path, 'train_opt.conf')
        pretrained_config = self.load_from_opt_record(pretrained_config_path)
        pretrained_config.isTrain = False 
        pretrained_config.gpu_ids = opt.gpu_ids 
        self.pretrained_encoder = UttSelfSuperviseModel(pretrained_config)
        self.pretrained_encoder.load_networks_cv(pretrained_path)
        self.pretrained_encoder.cuda()
        self.pretrained_encoder.eval()

        def post_process(self):
        def transform_key_for_parallel(state_dict):
            return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])

        if self.isTrain:
            print('[ Init ] Load parameters from pretrained encoder network')
            f = lambda x: transform_key_for_parallel(x)
            
            # ⭐ 只有不使用Mamba时才加载预训练LSTM权重
            if not self.use_mamba:
                self.netA.load_state_dict(f(self.pretrained_encoder.netA.state_dict()))
                self.netV.load_state_dict(f(self.pretrained_encoder.netV.state_dict()))
                self.netL.load_state_dict(f(self.pretrained_encoder.netL.state_dict()))
                self.netConA.load_state_dict(f(self.pretrained_encoder.netConA.state_dict()))
                self.netConV.load_state_dict(f(self.pretrained_encoder.netConV.state_dict()))
                self.netConL.load_state_dict(f(self.pretrained_encoder.netConL.state_dict()))
            else:
                print('[ Warning ] Using Mamba encoders, skipping pretrained LSTM weights loading')
                print('[ Info ] Mamba encoders will be trained from scratch')

    def load_from_opt_record(self, file_path):
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt

    def set_input(self, input):
        self.acoustic = acoustic = input['A_feat'].float().to(self.device)
        self.lexical = lexical = input['L_feat'].float().to(self.device)
        self.visual = visual = input['V_feat'].float().to(self.device)

        if 'missing_index' in input:
            self.missing_index = input['missing_index'].long().to(self.device)

        if self.isTrain:
            self.label = input['label'].to(self.device)
            self.missing_index = input['missing_index'].long().to(self.device) 
            self.A_miss_index = self.missing_index[:, 0].unsqueeze(1).unsqueeze(2)
            self.A_miss = acoustic * self.A_miss_index
            self.A_reverse = acoustic * -1 * (self.A_miss_index - 1)
            
            self.V_miss_index = self.missing_index[:, 1].unsqueeze(1).unsqueeze(2)
            self.V_miss = visual * self.V_miss_index
            self.V_reverse = visual * -1 * (self.V_miss_index - 1)
            
            self.L_miss_index = self.missing_index[:, 2].unsqueeze(1).unsqueeze(2)
            self.L_miss = lexical * self.L_miss_index
            self.L_reverse = lexical * -1 * (self.L_miss_index - 1)
            
            if self.opt.corpus_name == 'MOSI':
                self.label = self.label.unsqueeze(1)
        else:
            self.A_miss = acoustic
            self.V_miss = visual
            self.L_miss = lexical
            if 'label' in input:
                self.label = input['label'].to(self.device)
                if self.opt.corpus_name == 'MOSI':
                    self.label = self.label.unsqueeze(1)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # --- 1. 特征提取 ---
        self.feat_A_miss = self.netA(self.A_miss) 
        self.feat_L_miss = self.netL(self.L_miss)
        self.feat_V_miss = self.netV(self.V_miss)
        
        # 构造 h_real
        self.feat_fusion_miss = torch.cat([self.feat_A_miss, self.feat_L_miss, self.feat_V_miss], dim=-1)

        # --- 2. 提取共性特征 (Input Invariant) ---
        feat_A_con = self.netConA(self.A_miss)
        feat_L_con = self.netConL(self.L_miss)
        feat_V_con = self.netConV(self.V_miss)

        # =========================================================================
        # [核心改进] 软门控融合 (Soft Gating Fusion)
        # =========================================================================
        
        # 1. 获取 Prompt (兼容 DataParallel)
        if isinstance(self.netPrompts, nn.DataParallel):
            prompts_module = self.netPrompts.module
        else:
            prompts_module = self.netPrompts

        # 2. 逐模态计算 Alpha 并融合
        # Audio
        hard_mask_a = self.missing_index[:, 0].unsqueeze(1).float()
        # 【修正】使用 self.netgate_A
        alpha_a = self.netgate_A(feat_A_con, hard_mask_a) 
        curr_prompt_A = prompts_module.prompt_A.expand(feat_A_con.size(0), -1)
        feat_A_refined = feat_A_con * alpha_a + curr_prompt_A * (1 - alpha_a)

        # Text
        hard_mask_l = self.missing_index[:, 2].unsqueeze(1).float()
        # 【修正】使用 self.netgate_L
        alpha_l = self.netgate_L(feat_L_con, hard_mask_l)
        curr_prompt_L = prompts_module.prompt_L.expand(feat_L_con.size(0), -1)
        feat_L_refined = feat_L_con * alpha_l + curr_prompt_L * (1 - alpha_l)

        # Video
        hard_mask_v = self.missing_index[:, 1].unsqueeze(1).float()
        # 【修正】使用 self.netgate_V
        alpha_v = self.netgate_V(feat_V_con, hard_mask_v)
        curr_prompt_V = prompts_module.prompt_V.expand(feat_V_con.size(0), -1)
        feat_V_refined = feat_V_con * alpha_v + curr_prompt_V * (1 - alpha_v)

        # 3. 构造 Refined Query
        self.consistent_miss = torch.cat([feat_A_refined, feat_L_refined, feat_V_refined], dim=-1)

        # --- 3. 想象/重构 (Imagined) ---
        self.recon_fusion, _ = self.netAE(self.feat_fusion_miss, self.consistent_miss)

        # =========================================================================
        # 调用 FusionOpt
        # =========================================================================
        self.logits, _ = self.netFusionOpt(
            h_real=self.feat_fusion_miss, 
            h_imagined=self.recon_fusion, 
            h_invariant=self.consistent_miss
        )

        if self.opt.corpus_name != 'MOSI':
            self.pred = F.softmax(self.logits, dim=-1)
        else:
            self.pred = self.logits

        # for training 
        if self.isTrain:
            with torch.no_grad():
                self.T_embd_A = self.pretrained_encoder.netA(self.A_reverse)
                self.T_embd_L = self.pretrained_encoder.netL(self.L_reverse)
                self.T_embd_V = self.pretrained_encoder.netV(self.V_reverse)
                self.T_embds = torch.cat([self.T_embd_A, self.T_embd_L, self.T_embd_V], dim=-1)

                embd_A_consistent = self.pretrained_encoder.netConA(self.acoustic)
                embd_L_consistent = self.pretrained_encoder.netConL(self.lexical)
                embd_V_consistent = self.pretrained_encoder.netConV(self.visual)
                self.consistent = torch.cat([embd_A_consistent, embd_L_consistent, embd_V_consistent], dim=-1)

    def backward(self):
        # 分类损失
        self.loss_CE = self.ce_weight * self.criterion_ce(self.logits, self.label)
        # forward损失
        self.loss_mse = self.mse_weight * self.criterion_mse(self.T_embds, self.recon_fusion)
        # 共性特征损失
        self.loss_consistent = self.consistent_weight * self.criterion_mse(self.consistent, self.consistent_miss)
        
        # 综合损失
        loss = self.loss_CE + self.loss_mse + self.loss_consistent
        loss.backward()
        
        # 梯度裁剪
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

    def optimize_parameters(self, epoch):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()