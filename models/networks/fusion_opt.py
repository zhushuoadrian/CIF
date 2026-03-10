import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

try:
    from mamba_ssm import Mamba
except ImportError:
    from models.networks.mamba import Mamba

class OptimizedFusionClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5, d_state=32):
        """
        [True Mamba Fusion 终极进化版]
        摒弃累赘的交叉注意力，直接利用 Mamba 进行 3维特征序列空间的深度状态交互！
        """
        super().__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(dropout)
        
        self.has_mamba = True
        print(f"🌟 [True Mamba Fusion] 终极序列融合模式已开启！状态维度 d_state={d_state}")
        
        # Mamba 融合核心器
        self.mamba_layer = Mamba(
            d_model=input_dim, 
            d_state=d_state,   
            d_conv=4,          
            expand=2           
        )

        # 最终分类头
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, output_dim)
        )

    def forward(self, h_real, h_imagined, h_invariant):
        """
        Args:
            h_real:      [B, D]
            h_imagined:  [B, D]
            h_invariant: [B, D]
        """
        # ==========================================================
        # 核心创新：构建长度为 3 的多模态特征序列！
        # 顺序：真实特征 -> 想象补全特征 -> 不变性特征
        # ==========================================================
        # seq 维度: [B, 3, D]
        seq = torch.stack([h_real, h_imagined, h_invariant], dim=1)
        
        # ==========================================================
        # 状态空间融合：Mamba 的 SSM 机制会依次处理这3个特征，
        # 在处理到最后一个时，状态已经完美融合了前两者的信息！
        # ==========================================================
        # mamba_out 维度: [B, 3, D]
        mamba_out = self.mamba_layer(seq)
        
        # 取序列的最后一个状态（融合了所有信息的最终态）作为表征
        # feat 维度: [B, D]
        feat = mamba_out[:, -1, :]
        
        # 降维并分类
        feat = self.dropout(feat) 
        logits = self.classifier(feat)
        
        return logits, feat