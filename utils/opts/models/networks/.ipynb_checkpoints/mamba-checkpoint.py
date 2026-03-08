import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

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

    # ... (embd_attention, embd_maxpool, embd_last, embd_dense 方法保持和之前一模一样不变) ...
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