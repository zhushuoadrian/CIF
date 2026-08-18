import torch
import torch.nn as nn

class OptimizedFusionClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3, d_state=16):
        """
        [🧪 终极消融实验版本：w/o Optimized Fusion]
        同时移除 IGQF (交叉注意力) 和 Gated FFN (Sigmoid 门控)。
        退化为多模态融合中最基础的 Baseline：直接将观测特征与重构特征相加，不进行任何提纯与保护。
        """
        super().__init__()
        self.input_dim = input_dim
        
        print("⚠️ [Fusion] 启用终极消融模式：w/o Optimized Fusion (IGQF + Gate 均被移除)。")
        
        # 1. 基础特征空间对齐层 (保留此层以确保特征维度一致，但不再承担为注意力准备特征的复杂任务)
        self.feat_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # -------------------------------------------------------------
        # ✂️ 【消融点一】：彻底移除交叉注意力提纯机制 (Cross-Attention)
        # -------------------------------------------------------------
        
        # 2. 简单的归一化层 (替代了 Gated FFN 复杂的边界约束)
        self.norm1 = nn.LayerNorm(input_dim)
        
        # -------------------------------------------------------------
        # ✂️ 【消融点二】：彻底移除门控前馈网络 (Gate Projector)
        # -------------------------------------------------------------

        # 3. 分类器头 (保持原样，以接收最终特征并输出情绪得分)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, output_dim)
        )

    def forward(self, h_real, h_imagined, h_invariant):
        # -------------------------------------------------------------
        # 🚫 核心消融逻辑：
        # 无论传入什么样的“不变特征” (h_invariant)，在这个消融版中都彻底无视它！
        # 完全不利用它去洗涤噪声。
        # -------------------------------------------------------------
        
        # 1. 简单的特征空间对齐
        h_real_proj = self.feat_proj(h_real)
        h_imagined_proj = self.feat_proj(h_imagined)
        
        # 2. 暴力均值融合 (直接相加求平均，不经过任何 Attention 的过滤)
        fused_feat = (h_real_proj + h_imagined_proj) / 2.0
        
        # 3. 简单的层归一化 (直接归一化，没有任何 Sigmoid 的防爆挤压)
        final_feat = self.norm1(fused_feat)
            
        # 4. 直接送入分类器输出
        logits = self.classifier(final_feat)
        
        return logits, final_feat