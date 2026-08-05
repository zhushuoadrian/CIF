import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

# ====================================================================
# 1. 终极版：带残差连接的单模态 Mamba (用于处理文本等)
# ====================================================================
class MambaEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, embd_method='last', bidirectional=True): 
        super(MambaEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embd_method = embd_method
        self.bidirectional = bidirectional
        
        self.proj = nn.Linear(input_size, hidden_size)

        # 【新增】：前置归一化 Pre-Norm
        self.norm_fwd = nn.LayerNorm(hidden_size)
        self.mamba_fwd = Mamba(d_model=hidden_size, d_state=16, d_conv=4, expand=2)
        
        if self.bidirectional:
            self.norm_bwd = nn.LayerNorm(hidden_size)
            self.mamba_bwd = Mamba(d_model=hidden_size, d_state=16, d_conv=4, expand=2)
            self.fusion_layer = nn.Linear(hidden_size * 2, hidden_size)

        # 最终的归一化
        self.final_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.3)

        # 保留所有原有的池化层参数
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
        
        # 【核心修复】：前置 LayerNorm + Mamba + 残差连接 (+ x_proj)
        out_fwd = self.mamba_fwd(self.norm_fwd(x_proj)) + x_proj
        
        if self.bidirectional:
            x_reversed = torch.flip(x_proj, dims=[1])
            # 反向同样加上残差连接
            out_bwd = self.mamba_bwd(self.norm_bwd(x_reversed)) + x_reversed
            out_bwd = torch.flip(out_bwd, dims=[1])
            
            r_out = torch.cat([out_fwd, out_bwd], dim=-1)
            r_out = self.fusion_layer(r_out)
        else:
            r_out = out_fwd
            
        r_out = self.final_norm(r_out)
        r_out = self.dropout(r_out)
        
        embd = getattr(self, 'embd_' + self.embd_method)(r_out)
        return embd


# ====================================================================
# 2. 终极版：带残差连接的跨模态引导 Mamba (Res-TGC-Mamba)
# ====================================================================
class CrossMambaEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, guide_dim, embd_method='last', bidirectional=True):
        super(CrossMambaEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embd_method = embd_method
        self.bidirectional = bidirectional

        self.proj = nn.Linear(input_size, hidden_size)
        self.guide_proj = nn.Linear(guide_dim, hidden_size)

        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        # 【新增】：前置归一化 Pre-Norm
        self.norm_fwd = nn.LayerNorm(hidden_size)
        self.mamba_fwd = Mamba(d_model=hidden_size, d_state=16, d_conv=4, expand=2)
        
        if self.bidirectional:
            self.norm_bwd = nn.LayerNorm(hidden_size)
            self.mamba_bwd = Mamba(d_model=hidden_size, d_state=16, d_conv=4, expand=2)
            self.fusion_layer = nn.Linear(hidden_size * 2, hidden_size)

        self.final_norm = nn.LayerNorm(hidden_size)
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

    def forward(self, x, guide_x):
        x_proj = self.proj(x)            
        g_proj = self.guide_proj(guide_x) 

        attn_out, _ = self.cross_attn(query=x_proj, key=g_proj, value=g_proj)

        gate_weight = self.gate(torch.cat([x_proj, attn_out], dim=-1))
        x_fused = x_proj + gate_weight * attn_out

        # 【核心修复】：前置 LayerNorm + Mamba + 残差连接 (+ x_fused)
        out_fwd = self.mamba_fwd(self.norm_fwd(x_fused)) + x_fused
        
        if self.bidirectional:
            x_reversed = torch.flip(x_fused, dims=[1])
            # 反向同样加上残差连接
            out_bwd = self.mamba_bwd(self.norm_bwd(x_reversed)) + x_reversed
            out_bwd = torch.flip(out_bwd, dims=[1])
            
            r_out = torch.cat([out_fwd, out_bwd], dim=-1)
            r_out = self.fusion_layer(r_out)
        else:
            r_out = out_fwd
            
        r_out = self.final_norm(r_out)
        r_out = self.dropout(r_out)
        
        embd = getattr(self, 'embd_' + self.embd_method)(r_out)
        return embd