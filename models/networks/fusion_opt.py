import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

class OptimizedFusionClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3, d_state=16):
        """
        Args:
            input_dim: 输入特征维度 (例如 128)
            output_dim: 输出分类维度 (例如 4)
            dropout: Dropout 比率
            d_state: Mamba 的状态维度 (默认为 16)
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
        # 2. 序列建模模块 (Mamba / LSTM 自动切换)
        # ============================================================
        self.has_mamba = False
        try:
            from mamba_ssm import Mamba
            # 【情况 A：尝试引入 Mamba】
            print("🚀 [Fusion] Attempting to initialize Mamba...")
            self.mamba_layer = Mamba(
                d_model=input_dim,
                d_state=d_state,
                d_conv=4,
                expand=2
            )
            self.has_mamba = True
            print("✅ [Fusion] Successfully initialized Mamba.")
        except ImportError:
            # 【情况 B：环境没装好，回退到 LSTM】
            print("⚠️ [Fusion] Mamba_ssm not found. Fallback to LSTM.")
            self.lstm_layer = nn.LSTM(input_dim, input_dim, batch_first=True)
            self.has_mamba = False
        except Exception as e:
            # 【情况 C：其他 Mamba 报错，强制回退】
            print(f"⚠️ [Fusion] Mamba init failed: {e}. Fallback to LSTM.")
            self.lstm_layer = nn.LSTM(input_dim, input_dim, batch_first=True)
            self.has_mamba = False

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
        回归 Cross-Attention + Mamba Refinement 架构。
        """
        # 1. 构造 Query (Q) -> [B, 1, D]
        query = h_invariant.unsqueeze(1)
        
        # 2. 构造 Key/Value (K, V) -> [B, 2, D]
        kv = torch.stack([h_real, h_imagined], dim=1)
        
        # 3. 执行 Cross-Attention (先融合信息)
        # Q(共性) 去查询 K(真实+想象)，提取有用信息
        attn_output, _ = self.cross_attn(query, kv, kv) # [B, 1, D]
        
        # 4. 残差连接 + Norm
        feat = self.norm(attn_output + query) 
        
        # 5. Mamba 特征增强 (即使 Len=1，SSM 的门控机制依然有效)
        if self.has_mamba:
            feat = self.mamba_layer(feat)
        else:
            self.lstm_layer.flatten_parameters()
            feat, _ = self.lstm_layer(feat)
            
        # 6. 降维并分类
        feat = feat.squeeze(1) # [B, D]
        feat = self.dropout(feat)
        
        logits = self.classifier(feat)
        
        return logits, feat
   