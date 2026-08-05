import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

class OptimizedFusionClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3, d_state=16):
        """
        [增强版 SURE-Net 架构] 
        不变特征引导查询融合 (IGQF) + 门控前馈约束 (Gated FFN)
        """
        super().__init__()
        self.input_dim = input_dim
        
        # 为 Q, K, V 提供独立的特征空间对齐
        self.proj_q = nn.Sequential(nn.Linear(input_dim, input_dim), nn.LayerNorm(input_dim))
        self.proj_kv = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.cross_attn = MultiheadAttention(embed_dim=input_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(input_dim)
        
        # 门控前馈网络 
        self.gate_proj = nn.Linear(input_dim, input_dim * 2)
        self.norm2 = nn.LayerNorm(input_dim)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, output_dim)
        )

    def forward(self, h_real, h_imagined, h_invariant):
        # 1. 对 Query 进行独立的层归一化与投影，增强其作为“锚点”的稳定性
        Q = self.proj_q(h_invariant).unsqueeze(1) # [B, 1, D]
        
        # 2. 空间对齐与组装 (Key/Value) -> [B, 2, D]
        h_real_proj = self.proj_kv(h_real)
        h_imagined_proj = self.proj_kv(h_imagined)
        KV = torch.stack([h_real_proj, h_imagined_proj], dim=1)
        
        # 3. 跨注意力提纯
        attn_output, _ = self.cross_attn(Q, KV, KV) 
        
        # 4. 改进的残差连接：我们希望提纯后的特征依然保留真实的物理底色，
        # 因此以 h_real 作为残差基底，而不是用缺乏物理细节的 Q
        guided_feat = self.norm1(attn_output + h_real.unsqueeze(1)) 
        
        # 5. Gated FFN 边界约束机制
        gate_out = self.gate_proj(guided_feat)
        val, gate = gate_out.chunk(2, dim=-1)
        gated_feat = val * torch.sigmoid(gate)
        
        # 6. 第二次残差与归一化
        final_feat = self.norm2(gated_feat + guided_feat).squeeze(1) # [B, D]
            
        # 7. 分类输出
        logits = self.classifier(final_feat)
        
        return logits, final_feat