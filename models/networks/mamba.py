import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

# ====================================================================
# 1. 原始的双向 Mamba (用于处理单模态，比如文本)
# ====================================================================
class MambaEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, embd_method='last', bidirectional=True): # 默认开启双向
        super(MambaEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embd_method = embd_method
        self.bidirectional = bidirectional
        
        self.proj = nn.Linear(input_size, hidden_size)

        # 正向 Mamba
        self.mamba_fwd = Mamba(
            d_model=hidden_size, d_state=16, d_conv=4, expand=2,
        )
        
        # 反向 Mamba（如果开启双向的话）
        if self.bidirectional:
            self.mamba_bwd = Mamba(
                d_model=hidden_size, d_state=16, d_conv=4, expand=2,
            )
            # 因为双向合并后特征可能会变化，加一个线性层融合
            self.fusion_layer = nn.Linear(hidden_size * 2, hidden_size)

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.3)

        if self.embd_method == 'attention':
            self.attention_vector_weight = nn.Parameter(torch.Tensor(hidden_size, 1))
            self.attention_layer = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh())
            self.softmax = nn.Softmax(dim=1) 
            nn.init.xavier_uniform_(self.attention_vector_weight)
        elif self.embd_method == 'dense':
            self.dense_layer = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh())

    def embd_maxpool(self, r_out):
        in_feat = r_out.transpose(1, 2)
        embd = F.max_pool1d(in_feat, in_feat.size(2), in_feat.size(2))
        return embd.squeeze(-1)
        
    def embd_attention(self, r_out):
        hidden_reps = self.attention_layer(r_out)                      
        atten_weight = (hidden_reps @ self.attention_vector_weight)    
        atten_weight = self.softmax(atten_weight)                       
        sentence_vector = torch.sum(r_out * atten_weight, dim=1)        
        return sentence_vector

    def embd_last(self, r_out):
        return r_out[:, -1, :]

    def embd_dense(self, r_out):
        h_n = r_out[:, -1, :]
        return self.dense_layer(h_n)

    def forward(self, x):
        x_proj = self.proj(x)
        
        # 正向序列建模
        out_fwd = self.mamba_fwd(x_proj)
        
        if self.bidirectional:
            # 序列翻转 -> 过反向Mamba -> 再翻转回来
            x_reversed = torch.flip(x_proj, dims=[1])
            out_bwd = self.mamba_bwd(x_reversed)
            out_bwd = torch.flip(out_bwd, dims=[1])
            
            # 将正向和反向拼接，然后融合回 hidden_size
            r_out = torch.cat([out_fwd, out_bwd], dim=-1)
            r_out = self.fusion_layer(r_out)
        else:
            r_out = out_fwd
            
        r_out = self.norm(r_out)
        r_out = self.dropout(r_out)
        
        embd = getattr(self, 'embd_' + self.embd_method)(r_out)
        return embd


# ====================================================================
# 2. 【核心创新模块】跨模态引导的双向 Mamba (TGC-Mamba)
# ====================================================================
class CrossMambaEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, guide_dim, embd_method='last', bidirectional=True):
        super(CrossMambaEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embd_method = embd_method
        self.bidirectional = bidirectional

        # 1. 维度对齐投影 (分别对齐当前模态和引导模态)
        self.proj = nn.Linear(input_size, hidden_size)
        self.guide_proj = nn.Linear(guide_dim, hidden_size)

        # 2. 跨模态注意力与门控机制
        # 使用 MultiheadAttention 让引导模态(Text) 去提取 当前模态(Audio/Video) 中的关键信息
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)
        # 门控网络：决定融合多少跨模态信息
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        # 3. 状态空间模型 (Bi-Mamba)
        self.mamba_fwd = Mamba(d_model=hidden_size, d_state=16, d_conv=4, expand=2)
        if self.bidirectional:
            self.mamba_bwd = Mamba(d_model=hidden_size, d_state=16, d_conv=4, expand=2)
            self.fusion_layer = nn.Linear(hidden_size * 2, hidden_size)

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.3)

        # 池化方法兼容
        if self.embd_method == 'attention':
            self.attention_vector_weight = nn.Parameter(torch.Tensor(hidden_size, 1))
            self.attention_layer = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh())
            self.softmax = nn.Softmax(dim=1) 
            nn.init.xavier_uniform_(self.attention_vector_weight)
        elif self.embd_method == 'dense':
            self.dense_layer = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh())

    # 兼容同样的池化方法
    def embd_maxpool(self, r_out):
        in_feat = r_out.transpose(1, 2)
        embd = F.max_pool1d(in_feat, in_feat.size(2), in_feat.size(2))
        return embd.squeeze(-1)
        
    def embd_attention(self, r_out):
        hidden_reps = self.attention_layer(r_out)                      
        atten_weight = (hidden_reps @ self.attention_vector_weight)    
        atten_weight = self.softmax(atten_weight)                       
        sentence_vector = torch.sum(r_out * atten_weight, dim=1)        
        return sentence_vector

    def embd_last(self, r_out):
        return r_out[:, -1, :]

    def embd_dense(self, r_out):
        h_n = r_out[:, -1, :]
        return self.dense_layer(h_n)

    def forward(self, x, guide_x):
        # x: 当前待处理模态 (如音频A / 视觉V)
        # guide_x: 提供指导信息的模态 (如文本L)
        
        x_proj = self.proj(x)            # [batch, seq_len, hidden_size]
        g_proj = self.guide_proj(guide_x) # [batch, seq_len, hidden_size]

        # 【跨模态交互】：Query=当前模态, Key=Value=引导模态
        # 目的是让音频/视频去寻找与文本情感相关的帧
        attn_out, _ = self.cross_attn(query=x_proj, key=g_proj, value=g_proj)

        # 【动态门控融合】：控制跨模态信息注入的比例
        gate_weight = self.gate(torch.cat([x_proj, attn_out], dim=-1))
        x_fused = x_proj + gate_weight * attn_out

        # 【Mamba 序列建模】：对融合了文本提示的序列进行记忆提取
        out_fwd = self.mamba_fwd(x_fused)
        
        if self.bidirectional:
            x_reversed = torch.flip(x_fused, dims=[1])
            out_bwd = self.mamba_bwd(x_reversed)
            out_bwd = torch.flip(out_bwd, dims=[1])
            
            r_out = torch.cat([out_fwd, out_bwd], dim=-1)
            r_out = self.fusion_layer(r_out)
        else:
            r_out = out_fwd
            
        r_out = self.norm(r_out)
        r_out = self.dropout(r_out)
        
        embd = getattr(self, 'embd_' + self.embd_method)(r_out)
        return embd