import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import os
import json
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier, Fusion
from models.networks.autoencoder_2 import ResidualAE
from models.utils.config import OptConfig
from models.utt_self_supervise_model import UttSelfSuperviseModel

# =========================================================================
# 🌟 代理生成器 (Anti-Hallucination Proxy)
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
# 🌟 混合方差门控 (Hybrid Variational Gate) - [使用完整正常版]
# =========================================================================
class HybridVariationalGate(nn.Module):
    def __init__(self, input_dim, feat_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, input_dim // 2), nn.ReLU())
        self.alpha_head = nn.Linear(input_dim // 2, 1)
        self.logvar_head = nn.Linear(input_dim // 2, feat_dim)
        
        nn.init.constant_(self.alpha_head.bias, 1.0)
        nn.init.constant_(self.logvar_head.bias, -2.0)

    # 💡 清理了无用的 hard_mask 参数
    def forward(self, feature, original_feat):
        h = self.shared(feature)
        base_alpha = torch.sigmoid(self.alpha_head(h))
        log_var = self.logvar_head(h)
        mean_logvar = log_var.mean(dim=-1, keepdim=True)
        uncertainty = F.softplus(mean_logvar) + 1e-4
        
        temperature = 0.05 
        raw_alpha = base_alpha * torch.exp(-temperature * uncertainty)
        
        # 遵循控制变量法：保留完整的动态门控抑制
        alpha = raw_alpha 
        
        feat_norm = torch.norm(original_feat, dim=-1, keepdim=True)
        dynamic_mask = (feat_norm > 1e-5).float()
        
        alpha = alpha * dynamic_mask
            
        return alpha, log_var

# =========================================================================
# 🌟 模态协调器 (Modality Coordinator)
# =========================================================================
class ModalityCoordinator(nn.Module):
    def __init__(self, collab_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(collab_dim, collab_dim // 2), nn.ReLU(), nn.Linear(collab_dim // 2, 3))

    def forward(self, collab_feat):
        weights = torch.sigmoid(self.net(collab_feat)) * 2.0
        return weights

# =========================================================================
# 🌟 融合分类器 - 【必须使用完整增强版 SURE-Net 架构，防止双重消融】
# =========================================================================
class OptimizedFusionClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3, d_state=16):
        super().__init__()
        self.input_dim = input_dim
        
        # 为 Q, K, V 提供独立的特征空间对齐
        self.proj_q = nn.Sequential(nn.Linear(input_dim, input_dim), nn.LayerNorm(input_dim))
        self.proj_kv = nn.Sequential(nn.Linear(input_dim, input_dim), nn.GELU(), nn.Dropout(dropout))
        
        self.cross_attn = MultiheadAttention(embed_dim=input_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(input_dim)
        
        self.gate_proj = nn.Linear(input_dim, input_dim * 2)
        self.norm2 = nn.LayerNorm(input_dim)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, output_dim)
        )

    def forward(self, h_real, h_imagined, h_invariant):
        # 完整版：使用 h_invariant 作为洗涤锚点
        Q = self.proj_q(h_invariant).unsqueeze(1) 
        
        h_real_proj = self.proj_kv(h_real)
        h_imagined_proj = self.proj_kv(h_imagined)
        KV = torch.stack([h_real_proj, h_imagined_proj], dim=1)
        
        attn_output, _ = self.cross_attn(Q, KV, KV) 
        
        # 以 h_real 作为残差基底保留物理细节
        guided_feat = self.norm1(attn_output + h_real.unsqueeze(1)) 
        
        gate_out = self.gate_proj(guided_feat)
        val, gate = gate_out.chunk(2, dim=-1)
        gated_feat = val * torch.sigmoid(gate)
        
        final_feat = self.norm2(gated_feat + guided_feat).squeeze(1) 
            
        logits = self.classifier(final_feat)
        return logits, final_feat

# =========================================================================
# 🌟 主模型 (CIFMMINModel)
# =========================================================================
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
        parser.add_argument('--ce_weight', type=float, default=1.0)
        parser.add_argument('--mse_weight', type=float, default=1.0)
        parser.add_argument('--consistent_weight', type=float, default=1.0)
        parser.add_argument('--mamba_d_state', type=int, default=16)
        parser.add_argument('--image_dir', type=str, default='./consistent_image')
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.loss_names = ['CE', 'mse', 'consistent', 'alignment']
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
        
        if self.opt.corpus_name not in ['MOSI', 'SIMS', 'MOSEI']:
            self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        else:
            self.netC = Fusion(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate)

        if self.isTrain:
            self.load_pretrained_encoder(opt)
            
            if self.opt.corpus_name not in ['MOSI', 'SIMS', 'MOSEI']:
                self.criterion_ce = torch.nn.CrossEntropyLoss()
            else:
                self.criterion_ce = torch.nn.L1Loss()
                
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
            self.A_miss = self.acoustic
            self.V_miss = self.visual
            self.L_miss = self.lexical
            self.missing_index = torch.ones(batch_size, 3).long().to(self.device)
            self.A_miss_index = torch.ones(batch_size, 1, 1).float().to(self.device)
        else:
            mask_A = (torch.rand_like(self.acoustic) > missing_rate).float().to(self.device)
            mask_V = (torch.rand_like(self.visual) > missing_rate).float().to(self.device)
            mask_L = (torch.rand_like(self.lexical) > missing_rate).float().to(self.device)
            self.A_miss = self.acoustic * mask_A
            self.V_miss = self.visual * mask_V
            self.L_miss = self.lexical * mask_L
            self.missing_index = torch.ones(batch_size, 3).long().to(self.device)
            self.A_miss_index = mask_A

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
            self.A_miss = acoustic 
            self.V_miss = visual 
            self.L_miss = lexical 
            if self.opt.corpus_name in ['MOSI', 'SIMS', 'MOSEI']:
                self.label = self.label.unsqueeze(1).float()
        else:
            self.A_miss = acoustic
            self.V_miss = visual
            self.L_miss = lexical
            if 'label' in input:
                self.label = input['label'].to(self.device)
                if self.opt.corpus_name in ['MOSI', 'SIMS', 'MOSEI']:
                    self.label = self.label.unsqueeze(1).float()

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
        coord_module = self.netCoordinator.module if isinstance(self.netCoordinator, nn.DataParallel) else self.netCoordinator

        # ----------------------------------------------------
        # 🧪 【消融实验 1：w/o CMSP】
        # 获取 proxy 后直接强行清零，彻底阻断跨模态协同预期补偿
        # ----------------------------------------------------
        proxy_A_raw, proxy_V_raw, proxy_L_raw = proxy_module(collab_feat)
        proxy_A = torch.zeros_like(proxy_A_raw)
        proxy_V = torch.zeros_like(proxy_V_raw)
        proxy_L = torch.zeros_like(proxy_L_raw)
        # ----------------------------------------------------

        # 💡 已清理掉传给网关的冗余 hard_mask
        alpha_a, self.logvar_a = self.netgate_A(collab_feat, feat_A_con) 
        self.feat_A_refined = feat_A_con * alpha_a + proxy_A * (1 - alpha_a)

        alpha_l, self.logvar_l = self.netgate_L(collab_feat, feat_L_con)
        self.feat_L_refined = feat_L_con * alpha_l + proxy_L * (1 - alpha_l)

        alpha_v, self.logvar_v = self.netgate_V(collab_feat, feat_V_con)
        self.feat_V_refined = feat_V_con * alpha_v + proxy_V * (1 - alpha_v)

        self.consistent_miss_pre_weight = torch.cat([self.feat_A_refined, self.feat_L_refined, self.feat_V_refined], dim=-1)
        weights = coord_module(collab_feat)
        w_a, w_l, w_v = weights[:, 0].unsqueeze(1), weights[:, 1].unsqueeze(1), weights[:, 2].unsqueeze(1)

        self.consistent_miss = torch.cat([self.feat_A_refined * w_a, self.feat_L_refined * w_l, self.feat_V_refined * w_v], dim=-1)
        self.feat_fusion_miss = torch.cat([self.feat_A_miss * w_a, self.feat_L_miss * w_l, self.feat_V_miss * w_v], dim=-1)

        self.recon_fusion, _ = self.netAE(self.feat_fusion_miss, self.consistent_miss)

        # 🌟 修复点 1：阻断梯度的回流污染 (.detach())
        self.logits, self.H_final = self.netFusionOpt(
            h_real=self.feat_fusion_miss, 
            h_imagined=self.recon_fusion, 
            h_invariant=self.consistent_miss.detach()
        )

        if self.opt.corpus_name not in ['MOSI', 'SIMS', 'MOSEI']:
            self.pred = F.softmax(self.logits, dim=-1)
        else:
            self.pred = self.logits

        if self.isTrain:
            with torch.no_grad():
                self.T_embd_A = self.netA(self.acoustic).detach()
                self.T_embd_L = self.netL(self.lexical).detach()
                self.T_embd_V = self.netV(self.visual).detach()
                
                # 🌟 修复点 2：乘以协同权重对齐目标量级
                self.T_embds = torch.cat([self.T_embd_A * w_a, self.T_embd_L * w_l, self.T_embd_V * w_v], dim=-1)

                self.teacher_con_A = self.pretrained_encoder.netConA(self.acoustic)
                self.teacher_con_L = self.pretrained_encoder.netConL(self.lexical)
                self.teacher_con_V = self.pretrained_encoder.netConV(self.visual)
                self.consistent = torch.cat([self.teacher_con_A, self.teacher_con_L, self.teacher_con_V], dim=-1)

    def backward(self):
        self.loss_CE = self.ce_weight * self.criterion_ce(self.logits, self.label)
        self.loss_mse = self.mse_weight * self.criterion_mse(self.T_embds, self.recon_fusion)
        self.loss_consistent = (self.consistent_weight * 0.5) * self.criterion_mse(self.consistent_miss_pre_weight, self.consistent.detach())
        
        cos_sim_a = F.cosine_similarity(self.feat_A_refined, self.teacher_con_A.detach(), dim=-1).mean()
        cos_sim_v = F.cosine_similarity(self.feat_V_refined, self.teacher_con_V.detach(), dim=-1).mean()
        cos_sim_l = F.cosine_similarity(self.feat_L_refined, self.teacher_con_L.detach(), dim=-1).mean()
        
        self.loss_alignment = 0.05 * ( (1.0 - cos_sim_a) + (1.0 - cos_sim_v) + (1.0 - cos_sim_l) )
        
        loss = self.loss_CE + self.loss_mse + self.loss_consistent + self.loss_alignment
        loss.backward()
        for model in self.model_names: torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

    def optimize_parameters(self, epoch):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()