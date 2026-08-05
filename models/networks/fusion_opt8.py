import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

class OptimizedFusionClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3, d_state=16):
        """
        [The Golden SOTA] 
        不变特征引导查询融合 (IGQF) + LSTM 稳健边界约束
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
        self.norm_attn = nn.LayerNorm(input_dim)
        
        # 3. LSTM (提供绝佳的 Tanh/Sigmoid 边界约束与强正则化)
        print("✅ [Fusion] 启用最稳健的黄金架构：IGQF + LSTM 特征洗涤与边界约束。")
        self.lstm_layer = nn.LSTM(input_dim, input_dim, batch_first=True)
        self.norm_lstm = nn.LayerNorm(input_dim)

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
        
        # 4. 第一次残差与归一化
        guided_feat = self.norm_attn(attn_output + Q) 
        
        # 5. 走 LSTM 分支挖掘深层关系，利用其内部的门控进行极端的特征挤压约束
        self.lstm_layer.flatten_parameters()
        lstm_out, _ = self.lstm_layer(guided_feat)
        
        # 6. 第二次残差与归一化 
        final_feat = self.norm_lstm(lstm_out + guided_feat)
            
        # 7. 降维并分类
        final_feat = final_feat.squeeze(1) # [B, D]
        logits = self.classifier(final_feat)
        
        return logits, final_feat