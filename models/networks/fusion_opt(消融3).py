import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedFusionClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3, d_state=16):
        """
        🧪 [严格消融实验版本：w/o LSDM (无不变语义引导)]
        
        论文声明：移除潜空间一致性表征引导，将融合模块退化为标准的交叉注意力机制。
        实现逻辑：
        1. 严格控制变量：保留交叉注意力（Cross-Attention）和后置的 Gated FFN（门控前馈网络）。
        2. 核心消融点：将 Query (Q) 的来源由 h_invariant 替换为 h_real（即论文中的 h_obs），
           使其退化为没有理想锚点指引的普通交叉注意力检索。
        """
        super().__init__()
        self.input_dim = input_dim
        
        print("💡 [Fusion] 已启用严格消融模式：w/o LSDM（移除不变语义引导，退化为标准交叉注意力）。")
        
        # 1. 交叉注意力投影层 (对应论文公式 94, 95)
        # Q 的输入改为 h_real，维度为 input_dim (384)
        self.Wq = nn.Linear(input_dim, input_dim)
        # K 和 V 的输入是 [h_real || h_imagined] 级联，维度为 input_dim * 2 (768)
        self.Wk = nn.Linear(input_dim * 2, input_dim)
        self.Wv = nn.Linear(input_dim * 2, input_dim)
        
        self.scale = (input_dim) ** 0.5
        self.layer_norm1 = nn.LayerNorm(input_dim)
        
        # 2. 门控前馈网络 (Gated FFN) 投影层 (对应论文公式 100, 101)
        # 严格保留此组件以控制变量，仅消融 LSDM 的引导信号
        self.Wgate = nn.Linear(input_dim, input_dim * 2)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        
        # 3. 主干判别分类器头
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, output_dim)
        )

    def forward(self, h_real, h_imagined, h_invariant):
        """
        Args:
            h_real (Tensor): 观测联合特征 [Batch, input_dim] (即论文中的 h_obs)
            h_imagined (Tensor): 潜空间重构特征 [Batch, input_dim] (即论文中的 h_ima)
            h_invariant (Tensor): 第一阶段预训练的不变语义锚点 [Batch, input_dim] (即 h_inv)
        """
        # -------------------------------------------------------------
        # 🚫 核心消融逻辑：
        # 彻底孤立并无视传入的 h_invariant 变量！
        # 将 Query (Q) 的生成源头替换为原始观测特征 h_real
        # -------------------------------------------------------------
        
        # 1. 计算 Q, K, V (对应论文公式 94, 95 的退化版)
        Q = self.Wq(h_real)  # [Batch, input_dim] <- 关键消融点：用 h_real 替代了 h_invariant
        
        # K 和 V 依旧保持观测特征与重构特征的级联形式
        K_V_input = torch.cat([h_real, h_imagined], dim=-1)  # [Batch, input_dim * 2]
        K = self.Wk(K_V_input)  # [Batch, input_dim]
        V = self.Wv(K_V_input)  # [Batch, input_dim]
        
        # 2. 深度交叉注意力检索机制 (对应论文公式 97, 98)
        # 由于特征是话语级向量 [B, D]，此处执行样本/实例维度的交叉注意力矩阵乘法 (Q K^T)
        attn_scores = torch.matmul(Q, K.transpose(0, 1)) / self.scale  # [Batch, Batch]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [Batch, Batch]
        
        H_attn = torch.matmul(attn_weights, V)  # [Batch, input_dim]
        h_guided = self.layer_norm1(H_attn + Q)  # [Batch, input_dim]
        
        # 3. 门控前馈网络 (Gated FFN) 约束洗涤 (对应论文公式 100, 101)
        # 保留完整的动态滤噪逻辑，确保性能劣化完全归因于“失去了不变语义的引导”
        gate_projected = self.Wgate(h_guided)  # [Batch, input_dim * 2]
        v_gate, g = torch.chunk(gate_projected, 2, dim=-1)  # 切分为两个 [Batch, input_dim]
        
        final_feat = self.layer_norm2((v_gate * torch.sigmoid(g)) + h_guided)  # [Batch, input_dim]
            
        # 4. 送入分类器输出情感预测
        logits = self.classifier(final_feat)
        
        return logits, final_feat