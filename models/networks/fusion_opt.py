import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

class OptimizedFusionClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3, d_state=16):
        """
        [1.1072 SOTA 还原版]
        基于 Cross-Attention 和 LSTM 的融合层
        """
        super().__init__()
        self.input_dim = input_dim
        
        # ============================================================
        # 1. Cross-Attention 模块
        # ============================================================
        self.cross_attn = MultiheadAttention(embed_dim=input_dim, num_heads=4, dropout=dropout, batch_first=True)
        
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
        # ============================================================
        # 2. 序列建模模块 (强行锁定为当时的 LSTM 状态)
        # ============================================================
        self.has_mamba = False
        print("✅ [Fusion] 1.1072 还原模式：强制使用 LSTM 层进行特征增强。")
        self.lstm_layer = nn.LSTM(input_dim, input_dim, batch_first=True)

        # ============================================================
        # 3. 分类器头
        # ============================================================
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
            h_invariant: [B, D] -> Query
        """
        # 1. 构造 Query (Q) -> [B, 1, D]
        query = h_invariant.unsqueeze(1)
        
        # 2. 构造 Key/Value (K, V) -> [B, 2, D]
        kv = torch.stack([h_real, h_imagined], dim=1)
        
        # 3. 执行 Cross-Attention
        attn_output, _ = self.cross_attn(query, kv, kv) # [B, 1, D]
        
        # 4. 残差连接 + Norm
        feat = self.norm(attn_output + query) 
        
        # 5. 序列建模 (走 LSTM 分支)
        self.lstm_layer.flatten_parameters() # 加速
        feat, _ = self.lstm_layer(feat)
            
        # 6. 降维并分类
        feat = feat.squeeze(1) # [B, D]
        feat = self.dropout(feat) 
        
        logits = self.classifier(feat)
        
        return logits, feat