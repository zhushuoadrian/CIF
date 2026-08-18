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
from models.networks.fusion_opt import OptimizedFusionClassifier
from models.utils.config import OptConfig
from models.utt_self_supervise_model import UttSelfSuperviseModel

# =========================================================================
# 🌟 优化 3: 防毒幻觉的代理生成器 (Anti-Hallucination Proxy)
# 加入了 Dropout 和 LayerNorm，防止在 90% 缺失时凭空捏造极端噪声
# =========================================================================
class DynamicProxyGenerator(nn.Module):
    def __init__(self, collab_dim, embd_size_a, embd_size_v, embd_size_l):
        super().__init__()
        self.gen_A = nn.Sequential(
            nn.Linear(collab_dim, collab_dim // 2), 
            nn.LayerNorm(collab_dim // 2),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(collab_dim // 2, embd_size_a)
        )
        self.gen_V = nn.Sequential(
            nn.Linear(collab_dim, collab_dim // 2), 
            nn.LayerNorm(collab_dim // 2),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(collab_dim // 2, embd_size_v)
        )
        self.gen_L = nn.Sequential(
            nn.Linear(collab_dim, collab_dim // 2), 
            nn.LayerNorm(collab_dim // 2),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(collab_dim // 2, embd_size_l)
        )

    def forward(self, collab_feat):
        return self.gen_A(collab_feat), self.gen_V(collab_feat), self.gen_L(collab_feat)

# =========================================================================
# 🌟 优化 2: 残差保底门控 (Residual Variational Gate)
# 防止门控在 0% 缺失率下因“过度敏感”而误杀真实特征
# =========================================================================
class HybridVariationalGate(nn.Module):
    def __init__(self, input_dim, feat_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, input_dim // 2), nn.ReLU())
        self.alpha_head = nn.Linear(input_dim // 2, 1)
        self.logvar_head = nn.Linear(input_dim // 2, feat_dim)
        
        nn.init.constant_(self.alpha_head.bias, 1.0)
        nn.init.constant_(self.logvar_head.bias, -2.0)

    def forward(self, feature, hard_mask=None):
        h = self.shared(feature)
        base_alpha = torch.sigmoid(self.alpha_head(h))
        log_var = self.logvar_head(h)
        mean_logvar = log_var.mean(dim=-1, keepdim=True)
        uncertainty = F.softplus(mean_logvar) + 1e-4
        
        # 降低惩罚温度，并给予 20% 的强制保底流通量
        temperature = 0.05 
        raw_alpha = base_alpha * torch.exp(-temperature * uncertainty)
        alpha = 0.2 + 0.8 * raw_alpha 
        
        if hard_mask is not None:
            alpha = alpha * hard_mask
            
        return alpha, log_var

class ModalityCoordinator(nn.Module):
    def __init__(self, collab_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(collab_dim, collab_dim // 2), nn.ReLU(), nn.Linear(collab_dim // 2, 3))

    def forward(self, collab_feat):
        weights = torch.sigmoid(self.net(collab_feat)) * 2.0
        return weights


class CIFMMINModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim_a', type=int, default=130)
        parser.add_argument('--input_dim_l', type=int, default=1024)
        parser.add_argument('--input_dim_v', type=int, default=384)
        parser.add_argument('--embd_size_a', default=128, type=int)
        parser.add_argument('--embd_size_l', default=128, type=int)
        parser.add_argument('--embd_size_v', default=128, type=int)
        parser.add_argument('--embd_method_a', default='maxpool', type=str)
        parser.add_argument('--embd_method_v', default='maxpool', type=str)
        parser.add_argument('--AE_layers', type=str, default='128,64,32')
        parser.add_argument('--n_blocks', type=int, default=3)
        parser.add_argument('--cls_layers', type=str, default='128,128')
        parser.add_argument('--dropout_rate', type=float, default=0.3)
        parser.add_argument('--bn', action='store_true')
        parser.add_argument('--pretrained_path', type=str)
        parser.add_argument('--pretrained_consistent_path', type=str)
        parser.add_argument('--ce_weight', type=float, default=1.0)
        parser.add_argument('--mse_weight', type=float, default=1.0)
        parser.add_argument('--cycle_weight', type=float, default=1.0)
        parser.add_argument('--consistent_weight', type=float, default=1.0)
        parser.add_argument('--share_weight', action='store_true')
        parser.add_argument('--image_dir', type=str, default='./consistent_image')
        parser.add_argument('--mamba_d_state', type=int, default=16)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.loss_names = ['CE', 'mse', 'consistent', 'KL_align']
        self.model_names = ['C', 'AE', 'A', 'ConA', 'L', 'ConL', 'V', 'ConV', 'FusionOpt', 'DynamicProxy', 'gate_A', 'gate_L', 'gate_V', 'Coordinator'] 

        self.netA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        self.netConA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        self.netL = TextCNN(opt.input_dim_l, opt.embd_size_l, dropout=0.5)
        self.netConL = LSTMEncoder(opt.input_dim_l, opt.embd_size_l)
        self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
        self.netConV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
        
        AE_layers = list(map(lambda x: int(x), opt.AE_layers.split(',')))
        AE_input_dim = opt.embd_size_a + opt.embd_size_v + opt.embd_size_l
        self.netAE = ResidualAE(AE_layers, opt.n_blocks, AE_input_dim, dropout=0, use_bn=False)
        
        gate_collab_dim = opt.embd_size_a + opt.embd_size_l + opt.embd_size_v
        cls_input_size = opt.embd_size_a + opt.embd_size_v + opt.embd_size_l
        
        mamba_state = getattr(opt, 'mamba_d_state', 16)
        self.netFusionOpt = OptimizedFusionClassifier(input_dim=cls_input_size, output_dim=opt.output_dim, dropout=opt.dropout_rate, d_state=mamba_state)
        
        self.netDynamicProxy = DynamicProxyGenerator(gate_collab_dim, opt.embd_size_a, opt.embd_size_v, opt.embd_size_l)
        
        self.netgate_A = HybridVariationalGate(gate_collab_dim, opt.embd_size_a)
        self.netgate_L = HybridVariationalGate(gate_collab_dim, opt.embd_size_l)
        self.netgate_V = HybridVariationalGate(gate_collab_dim, opt.embd_size_v)
        
        self.netCoordinator = ModalityCoordinator(gate_collab_dim)

        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        if self.opt.corpus_name != 'MOSI':
            self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        else:
            self.netC = Fusion(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate)

        if self.isTrain:
            self.load_pretrained_encoder(opt)
            self.criterion_ce = torch.nn.CrossEntropyLoss() if self.opt.corpus_name != 'MOSI' else torch.nn.MSELoss()
            self.criterion_mse = torch.nn.MSELoss()
            
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim
            self.ce_weight = opt.ce_weight
            self.mse_weight = opt.mse_weight
            self.consistent_weight = opt.consistent_weight
        else:
            self.load_pretrained_encoder(opt)

        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir): os.mkdir(self.save_dir)

    def load_pretrained_encoder(self, opt):
        pretrained_path = os.path.join(opt.pretrained_path, str(opt.cvNo))
        pretrained_config_path = os.path.join(opt.pretrained_path, 'train_opt.conf')
        pretrained_config = self.load_from_opt_record(pretrained_config_path)
        pretrained_config.isTrain = False 
        pretrained_config.gpu_ids = opt.gpu_ids 
        self.pretrained_encoder = UttSelfSuperviseModel(pretrained_config)
        self.pretrained_encoder.load_networks_cv(pretrained_path)
        self.pretrained_encoder.cuda()
        self.pretrained_encoder.eval()

    def load_from_opt_record(self, file_path):
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt

    def apply_random_missing(self, missing_rate=0.0):
        batch_size = self.acoustic.size(0)
        if missing_rate <= 0.0:
            mask_A = torch.ones(batch_size).to(self.device)
            mask_V = torch.ones(batch_size).to(self.device)
            mask_L = torch.ones(batch_size).to(self.device)
        else:
            mask_A = (torch.rand(batch_size) > missing_rate).float().to(self.device)
            mask_V = (torch.rand(batch_size) > missing_rate).float().to(self.device)
            mask_L = (torch.rand(batch_size) > missing_rate).float().to(self.device)

        self.missing_index = torch.stack([mask_A, mask_V, mask_L], dim=1).long()
        self.A_miss_index = self.missing_index[:, 0].unsqueeze(1).unsqueeze(2)
        self.A_miss = self.acoustic * self.A_miss_index
        self.V_miss_index = self.missing_index[:, 1].unsqueeze(1).unsqueeze(2)
        self.V_miss = self.visual * self.V_miss_index
        self.L_miss_index = self.missing_index[:, 2].unsqueeze(1).unsqueeze(2)
        self.L_miss = self.lexical * self.L_miss_index

    def set_input(self, input):
        self.acoustic = acoustic = input['A_feat'].float().to(self.device)
        self.lexical = lexical = input['L_feat'].float().to(self.device)
        self.visual = visual = input['V_feat'].float().to(self.device)
        
        batch_size = self.acoustic.size(0)

        if 'missing_index' in input:
            self.missing_index = input['missing_index'].long().to(self.device)
        else:
            self.missing_index = torch.ones(batch_size, 3).long().to(self.device)

        if self.isTrain:
            self.label = input['label'].to(self.device)
            
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
        self.feat_A_miss = self.netA(self.A_miss) 
        self.feat_L_miss = self.netL(self.L_miss)
        self.feat_V_miss = self.netV(self.V_miss)

        feat_A_con = self.netConA(self.A_miss)
        feat_L_con = self.netConL(self.L_miss)
        feat_V_con = self.netConV(self.V_miss)
        
        self.feat_A_con = feat_A_con
        self.feat_L_con = feat_L_con
        self.feat_V_con = feat_V_con

        collab_feat = torch.cat([feat_A_con, feat_L_con, feat_V_con], dim=-1)
        
        proxy_module = self.netDynamicProxy.module if isinstance(self.netDynamicProxy, nn.DataParallel) else self.netDynamicProxy
        coord_module = self.netCoordinator.module if isinstance(self.netDynamicProxy, nn.DataParallel) else self.netCoordinator

        proxy_A, proxy_V, proxy_L = proxy_module(collab_feat)

        hard_mask_a = self.missing_index[:, 0].unsqueeze(1).float()
        alpha_a, self.logvar_a = self.netgate_A(collab_feat, hard_mask_a) 
        feat_A_refined = feat_A_con * alpha_a + proxy_A * (1 - alpha_a)

        hard_mask_l = self.missing_index[:, 2].unsqueeze(1).float()
        alpha_l, self.logvar_l = self.netgate_L(collab_feat, hard_mask_l)
        feat_L_refined = feat_L_con * alpha_l + proxy_L * (1 - alpha_l)

        hard_mask_v = self.missing_index[:, 1].unsqueeze(1).float()
        alpha_v, self.logvar_v = self.netgate_V(collab_feat, hard_mask_v)
        feat_V_refined = feat_V_con * alpha_v + proxy_V * (1 - alpha_v)

        self.consistent_miss_pre_weight = torch.cat([feat_A_refined, feat_L_refined, feat_V_refined], dim=-1)
        weights = coord_module(collab_feat)
        w_a, w_l, w_v = weights[:, 0].unsqueeze(1), weights[:, 1].unsqueeze(1), weights[:, 2].unsqueeze(1)

        self.consistent_miss = torch.cat([feat_A_refined * w_a, feat_L_refined * w_l, feat_V_refined * w_v], dim=-1)
        self.feat_fusion_miss = torch.cat([self.feat_A_miss * w_a, self.feat_L_miss * w_l, self.feat_V_miss * w_v], dim=-1)

        self.recon_fusion, _ = self.netAE(self.feat_fusion_miss, self.consistent_miss)

        self.logits, _ = self.netFusionOpt(h_real=self.feat_fusion_miss, h_imagined=self.recon_fusion, h_invariant=self.consistent_miss)

        self.pred = F.softmax(self.logits, dim=-1) if self.opt.corpus_name != 'MOSI' else self.logits

        if self.isTrain:
            with torch.no_grad():
                self.T_embd_A = self.pretrained_encoder.netA(self.A_reverse)
                self.T_embd_L = self.pretrained_encoder.netL(self.L_reverse)
                self.T_embd_V = self.pretrained_encoder.netV(self.V_reverse)
                self.T_embds = torch.cat([self.T_embd_A, self.T_embd_L, self.T_embd_V], dim=-1)
                
                self.consistent = torch.cat([
                    self.pretrained_encoder.netConA(self.acoustic), 
                    self.pretrained_encoder.netConL(self.lexical), 
                    self.pretrained_encoder.netConV(self.visual)
                ], dim=-1)

    def kl_divergence(self, mu1, logvar1, mu2, logvar2):
        return 0.5 * torch.mean(
            torch.sum(logvar2 - logvar1 + (torch.exp(logvar1) + (mu1 - mu2)**2) / (torch.exp(logvar2) + 1e-6) - 1, dim=-1)
        )

    def backward(self):
        self.loss_CE = self.ce_weight * self.criterion_ce(self.logits, self.label)
        self.loss_mse = self.mse_weight * self.criterion_mse(self.T_embds, self.recon_fusion)
        self.loss_consistent = (self.consistent_weight * 0.5) * self.criterion_mse(self.consistent_miss_pre_weight, self.consistent)
        
        # 🌟 优化 1: 退火 KL 对齐权重
        # 权重从 0.1 降至 0.05，减轻强制对齐对分类尖锐度的磨损
        loss_kl_a_l = self.kl_divergence(self.feat_A_con, self.logvar_a, self.feat_L_con.detach(), self.logvar_l.detach())
        loss_kl_v_l = self.kl_divergence(self.feat_V_con, self.logvar_v, self.feat_L_con.detach(), self.logvar_l.detach())
        self.loss_KL_align = 0.05 * (loss_kl_a_l + loss_kl_v_l)
        
        loss = self.loss_CE + self.loss_mse + self.loss_consistent + self.loss_KL_align
        loss.backward()
        for model in self.model_names: torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

    def optimize_parameters(self, epoch):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()