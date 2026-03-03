import torch
import torch.nn as nn

# ==========================================================
# [新增] 手动实现 RMSNorm 以兼容旧版 PyTorch
# ==========================================================
try:
    # 尝试导入官方的 RMSNorm (PyTorch 2.1+)
    from torch.nn import RMSNorm
except ImportError:
    # 如果失败，使用手动实现
    class RMSNorm(nn.Module):
        def __init__(self, d_model: int, eps: float = 1e-5):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(d_model))

        def forward(self, x):
            output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return output * self.weight

# ==========================================================
# Mamba 导入检查
# ==========================================================
try:
    from mamba_ssm import Mamba
except ImportError:
    print("Warning: mamba_ssm not installed. Please install it via pip.")
    Mamba = None

class BiMambaEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2, dropout=0.1):
        """
        Args:
            input_dim: 输入特征维度 (如 130, 768)
            output_dim: 输出特征维度 (如 128)
            d_state: SSM 状态维度 (Mamba 默认为 16)
        """
        super().__init__()
        
        # 1. 投影层
        self.proj_in = nn.Linear(input_dim, output_dim)
        
        # 2. 双向 Mamba (Bi-Directional)
        self.mamba_fwd = Mamba(
            d_model=output_dim, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )
        self.mamba_bwd = Mamba(
            d_model=output_dim, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )
        
        # 3. 归一化 (使用上面定义的 RMSNorm)
        # 拼接后维度是 2 * output_dim
        self.norm = RMSNorm(output_dim * 2)
        
        # 4. 输出投影
        self.proj_out = nn.Linear(output_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Tanh() 

    def forward(self, x):
        """
        Input: [Batch, Seq_Len, Input_Dim]
        Output: [Batch, Output_Dim] (句子级向量)
        """
        # 线性投影
        x = self.proj_in(x) # [B, L, D]
        
        # === 双向处理 ===
        x_fwd = self.mamba_fwd(x)
        
        # 反向
        x_bwd_in = torch.flip(x, dims=[1])
        x_bwd_out = self.mamba_bwd(x_bwd_in)
        x_bwd = torch.flip(x_bwd_out, dims=[1])
        
        # 拼接
        x_bi = torch.cat([x_fwd, x_bwd], dim=-1) # [B, L, 2*D]
        
        # 归一化
        x_bi = self.norm(x_bi)
        x_bi = self.dropout(x_bi)
        
        # 融合双向特征
        feat_seq = self.proj_out(x_bi) # [B, L, D]
        
        # === 池化 (Pooling) ===
        feat_sent = torch.max(feat_seq, dim=1)[0] # [B, D]
        
        return self.act(feat_sent)