import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

class OptimizedFusionClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3, d_state=16):
        """
        [重构版 SOTA 架构] 
        不变特征引导查询融合 (IGQF) + 门控前馈约束 (Gated FFN)
        """
        super().__init__()
        self.input_dim = input_dim
        
        # 1. IGQF 特征空间对齐层 (将特征映射到干净空间)
        self.feat_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. 交叉注意力模块 (Cross-Attention)
        self.cross_attn = MultiheadAttention(embed_dim=input_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(input_dim)
        
        # 3. 门控前馈网络 (代替无意义的单步 LSTM，提供稳健的数值边界约束)
        print("✅ [Fusion] 启用重构架构：IGQF + Gated FFN 特征洗涤与边界约束。")
        self.gate_proj = nn.Linear(input_dim, input_dim * 2)
        self.norm2 = nn.LayerNorm(input_dim)

        # 4. 分类器头
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, output_dim)
        )

    def forward(self, h_real, h_imagined, h_invariant):
        # 1. 确立绝对权威 (Query) -> [B, 1, D]
        Q = h_invariant.unsqueeze(1)
        
        # 2. 空间对齐与组装 (Key/Value) -> [B, 2, D]
        h_real_proj = self.feat_proj(h_real)
        h_imagined_proj = self.feat_proj(h_imagined)
        KV = torch.stack([h_real_proj, h_imagined_proj], dim=1)
        
        # 3. 跨注意力提纯 (用纯净的 Q 洗涤含有重构噪声的 KV)
        attn_output, _ = self.cross_attn(Q, KV, KV) 
        
        # 4. 第一次残差与归一化 -> [B, 1, D]
        guided_feat = self.norm1(attn_output + Q) 
        
        # 5. Gated FFN 边界约束机制 (提取 Value 并用 Gate 进行 Sigmoid 挤压)
        gate_out = self.gate_proj(guided_feat)
        val, gate = gate_out.chunk(2, dim=-1)
        gated_feat = val * torch.sigmoid(gate)
        
        # 6. 第二次残差与归一化，随后降维 
        final_feat = self.norm2(gated_feat + guided_feat).squeeze(1) # [B, D]
            
        # 7. 分类输出
        logits = self.classifier(final_feat)
        
        return logits, final_feat