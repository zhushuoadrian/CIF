import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

class OptimizedFusionClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3, d_state=16):
        """
        🧪 [严格消融实验版本：w/o LSDM]
        保持与增强版 SURE-Net 完全一致的网络架构 (MultiheadAttention + Gated FFN)，
        但切断不变语义 h_invariant 的引导，将其退化为普通的交叉注意力。
        """
        super().__init__()
        self.input_dim = input_dim
        
        print("💡 [Fusion] 已启用严格消融模式：w/o LSDM (保持网络同构，但切断不变语义引导)。")
        
        # 1. 保持与原模型同构的特征投影层
        self.proj_q = nn.Sequential(nn.Linear(input_dim, input_dim), nn.LayerNorm(input_dim))
        self.proj_kv = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. 采用与原模型完全一致的标准多头注意力 (彻底解决 Batch-wise 污染 Bug)
        self.cross_attn = MultiheadAttention(embed_dim=input_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(input_dim)
        
        # 3. 门控前馈网络保持不变 
        self.gate_proj = nn.Linear(input_dim, input_dim * 2)
        self.norm2 = nn.LayerNorm(input_dim)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, output_dim)
        )

    def forward(self, h_real, h_imagined, h_invariant):
        # ==========================================================
        # 🚫 核心消融点：彻底孤立并无视传入的 h_invariant 变量！
        # 强制使用 h_real (包含残缺噪声的观测特征) 来生成 Query 锚点
        # ==========================================================
        Q = self.proj_q(h_real).unsqueeze(1) # [Batch, 1, Dim]
        
        # 空间对齐与组装 (Key/Value) -> [Batch, 2, Dim]
        h_real_proj = self.proj_kv(h_real)
        h_imagined_proj = self.proj_kv(h_imagined)
        KV = torch.stack([h_real_proj, h_imagined_proj], dim=1)
        
        # 跨注意力提纯 (此时 Q 也是不纯净的，洗涤效果将大打折扣，以此证明 LSDM 的必要性)
        attn_output, _ = self.cross_attn(Q, KV, KV) 
        
        # 第一次残差与归一化
        guided_feat = self.norm1(attn_output + h_real.unsqueeze(1)) 
        
        # Gated FFN 边界约束机制
        gate_out = self.gate_proj(guided_feat)
        val, gate = gate_out.chunk(2, dim=-1)
        gated_feat = val * torch.sigmoid(gate)
        
        # 第二次残差、归一化与降维
        final_feat = self.norm2(gated_feat + guided_feat).squeeze(1) # [Batch, Dim]
            
        # 分类输出
        logits = self.classifier(final_feat)
        
        return logits, final_feat