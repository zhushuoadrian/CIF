import torch
import os
import json
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.lstm import LSTMEncoder  # <--- 已经为您改回标准的 LSTM
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier, Fusion
from models.networks.shared import SharedEncoder
from models.utils import CMD
from einops import rearrange, repeat, reduce
from torch import einsum

def masked_mean(t, mask, dim=1, eps=1e-6):
    t = t.masked_fill(~mask, 0.)
    return t.sum(dim=dim) / mask.sum(dim=dim).clamp(min=eps)

def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device=device)
    j_range = torch.arange(j, device=device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    return rearrange(t.masked_select(diag_mask), '(b d) -> b d', d=num_diag_el)

def log(t, eps=1e-20): return torch.log(t + eps)
def l2norm(t): return F.normalize(t, dim=-1)

class UttSelfSuperviseModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='visual input dim')
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'])
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'])
        parser.add_argument('--cls_layers', type=str, default='128,128')
        parser.add_argument('--dropout_rate', type=float, default=0.3)
        parser.add_argument('--bn', action='store_true')
        parser.add_argument('--modality', type=str, default='AVL')
        parser.add_argument('--image_dir', type=str, default='./consistent_image')
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.loss_names = ['TA', 'TV', 'VA', 'CE']
        self.modality = opt.modality
        self.model_names = ['SharedA', 'SharedV', 'SharedT', "C"]
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        cls_input_size = opt.embd_size_a * int("A" in self.modality) + \
                         opt.embd_size_v * int("V" in self.modality) + \
                         opt.embd_size_l * int("L" in self.modality)
        
        self.netSharedV = SharedEncoder(opt)
        self.netSharedA = SharedEncoder(opt)
        self.netSharedT = SharedEncoder(opt)
        
        if self.opt.corpus_name != 'MOSI':
            self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        else:
            self.netC = Fusion(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate)

        self.temperature = torch.nn.Parameter(torch.tensor(1.))
        self.batch_size = opt.batch_size
        
        # [关键修复]：全部改回纯正的 LSTM
        if 'A' in self.modality:
            self.model_names.extend(['A', 'ConA'])
            self.netA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
            self.netConA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)

        if 'L' in self.modality:
            self.model_names.extend(['L', 'ConL'])
            self.netL = TextCNN(opt.input_dim_l, opt.embd_size_l)
            self.netConL = LSTMEncoder(opt.input_dim_l, opt.embd_size_l)

        if 'V' in self.modality:
            self.model_names.extend(['V', 'ConV'])
            self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, embd_method=opt.embd_method_v)
            self.netConV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, embd_method=opt.embd_method_v)

        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss() if self.opt.corpus_name != 'MOSI' else torch.nn.MSELoss()
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim

        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir): os.mkdir(self.save_dir)

    def set_input(self, input):
        if 'A' in self.modality: self.acoustic = input['A_feat'].float().to(self.device)
        if 'L' in self.modality: self.lexical = input['L_feat'].float().to(self.device)
        if 'V' in self.modality: self.visual = input['V_feat'].float().to(self.device)
        self.label = input['label'].to(self.device)
        if self.opt.corpus_name == 'MOSI': self.label = self.label.unsqueeze(1)

    def forward(self):
        final_embd, final_shared = [], []
        
        # [关键修复]：去除多余的 lexical 引导参数，恢复 LSTM 单一输入
        if 'A' in self.modality:
            self.feat_A = self.netA(self.acoustic)
            feat_ConA = self.netConA(self.acoustic)
            self.feat_shared_A = self.netSharedA(feat_ConA)
            final_embd.append(self.feat_A)
            final_shared.append(feat_ConA)

        if 'L' in self.modality:
            self.feat_L = self.netL(self.lexical)
            feat_ConL = self.netConL(self.lexical)
            self.feat_shared_T = self.netSharedT(feat_ConL)
            final_embd.append(self.feat_L)
            final_shared.append(feat_ConL)

        if 'V' in self.modality:
            self.feat_V = self.netV(self.visual)
            feat_ConV = self.netConV(self.visual)
            self.feat_shared_V = self.netSharedV(feat_ConV)
            final_embd.append(self.feat_V)
            final_shared.append(feat_ConV)

        self.feat = torch.cat(final_embd, dim=-1)
        self.logits, self.ef_fusion_feat = self.netC(self.feat)
        self.pred = F.softmax(self.logits, dim=-1) if self.opt.corpus_name != "MOSI" else self.logits

    def backward(self):
        num_batch_texts = 1
        temp = self.temperature.exp()
        temperature = torch.nn.Parameter(torch.tensor(1e-10))
        
        self.feat_shared_V = rearrange(self.feat_shared_V, '(m b) ... -> m b ...', m=num_batch_texts)
        self.feat_shared_T = rearrange(self.feat_shared_T, '(m b) ... -> m b ...', m=num_batch_texts)
        self.feat_shared_A = rearrange(self.feat_shared_A, '(m b) ... -> m b ...', m=num_batch_texts)

        self.feat_shared_T, self.feat_shared_V = map(l2norm, (self.feat_shared_T, self.feat_shared_V))
        self.feat_shared_T, self.feat_shared_A = map(l2norm, (self.feat_shared_T, self.feat_shared_A))
        self.feat_shared_V, self.feat_shared_A = map(l2norm, (self.feat_shared_V, self.feat_shared_A))

        self.text_to_image = einsum('m t d, n i d -> m n t i', self.feat_shared_T, self.feat_shared_V) * temp
        self.image_to_text = rearrange(self.text_to_image, '... t i -> ... i t')
        self.text_to_audio = einsum('m t d, n i d -> m n t i', self.feat_shared_T, self.feat_shared_A) * temp
        self.audio_to_text = rearrange(self.text_to_audio, '... t i -> ... i t')
        self.image_to_audio = einsum('m t d, n i d -> m n t i', self.feat_shared_V, self.feat_shared_A) * temp
        self.audio_to_image = rearrange(self.image_to_audio, '... t i -> ... i t')

        self.text_to_image, self.image_to_text = map(lambda t: rearrange(t, 'm n ... -> (m n) ...'), (self.text_to_image, self.image_to_text))
        self.text_to_audio, self.audio_to_text = map(lambda t: rearrange(t, 'm n ... -> (m n) ...'), (self.text_to_audio, self.audio_to_text))
        self.image_to_audio, self.audio_to_image = map(lambda t: rearrange(t, 'm n ... -> (m n) ...'), (self.image_to_audio, self.audio_to_image))

        text_to_image_exp, image_to_text_exp = map(torch.exp, (self.text_to_image, self.image_to_text))
        text_to_audio_exp, audio_to_text_exp = map(torch.exp, (self.text_to_audio, self.audio_to_text))
        image_to_audio_exp, audio_to_image_exp = map(torch.exp, (self.image_to_audio, self.audio_to_image))

        text_to_image_pos, image_to_text_pos = map(matrix_diag, (text_to_image_exp, image_to_text_exp))
        text_to_audio_pos, audio_to_text_pos = map(matrix_diag, (text_to_audio_exp, audio_to_text_exp))
        image_to_audio_pos, audio_to_image_pos = map(matrix_diag, (image_to_audio_exp, audio_to_image_exp))

        pos_mask = torch.eye(self.lexical.shape[0], device=self.device, dtype=torch.bool)
        text_to_image_exp, image_to_text_exp = map(lambda t: t.masked_fill(pos_mask, 0.), (text_to_image_exp, image_to_text_exp))
        text_to_audio_exp, audio_to_text_exp = map(lambda t: t.masked_fill(pos_mask, 0.), (text_to_audio_exp, audio_to_text_exp))
        image_to_audio_exp, audio_to_image_exp = map(lambda t: t.masked_fill(pos_mask, 0.), (image_to_audio_exp, audio_to_image_exp))

        text_to_image_denom, image_to_text_denom = map(lambda t: t.sum(dim=-1), (text_to_image_exp, image_to_text_exp))
        text_to_audio_denom, audio_to_text_denom = map(lambda t: t.sum(dim=-1), (text_to_audio_exp, audio_to_text_exp))
        image_to_audio_denom, audio_to_image_denom = map(lambda t: t.sum(dim=-1), (image_to_audio_exp, audio_to_image_exp))

        text_to_image_loss = -log(text_to_image_pos / (text_to_image_denom + temperature)).mean(dim=-1)
        image_to_text_loss = -log(image_to_text_pos / (image_to_text_denom + temperature)).mean(dim=-1)
        text_to_audio_loss = -log(text_to_audio_pos / (text_to_audio_denom + temperature)).mean(dim=-1)
        audio_to_text_loss = -log(audio_to_text_pos / (audio_to_text_denom + temperature)).mean(dim=-1)
        image_to_audio_loss = -log(image_to_audio_pos / (image_to_audio_denom + temperature)).mean(dim=-1)
        audio_to_image_loss = -log(audio_to_image_pos / (audio_to_image_denom + temperature)).mean(dim=-1)

        cl_tv_loss = ((text_to_image_loss + image_to_text_loss) / 2)[0]
        cl_ta_loss = ((text_to_audio_loss + audio_to_text_loss) / 2)[0]
        cl_va_loss = ((image_to_audio_loss + audio_to_image_loss) / 2)[0]

        self.loss_TA = cl_ta_loss * 0.25
        self.loss_TV = cl_tv_loss * 0.25
        self.loss_VA = cl_va_loss * 0.5
        self.loss_CE = self.criterion_ce(self.logits, self.label)
        
        loss = self.loss_CE + self.loss_TV + self.loss_TA + self.loss_VA
        loss.backward()
        for model in self.model_names: torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 0.5)

    def optimize_parameters(self, epoch):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()