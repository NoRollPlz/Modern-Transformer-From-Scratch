"""
===================================================================================
项目名称: CLUE TNEWS 新闻分类竞赛实战 (消融实验与架构演进篇)
文件名称: model_v3.py
模块功能: V3 进阶模型 —— 包含 [CLS] Token、SwiGLU 门控、RMSNorm 三大可插拔架构
===================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_models.model_v1 import TokenEmbedding, SinusoidalPositionalEncoding
from train_models.model_v2 import AttentionPooling


# =================================================================================
# 架构组件 1: 现代大模型标配 RMSNorm (均方根归一化)
# =================================================================================
class RMSNorm(nn.Module):
    """
    【架构技术细节: RMSNorm (Root Mean Square Normalization)】
    
    数学公式:
        RMS(x) = sqrt( mean(x^2) + eps )
        y = (x / RMS(x)) * weight
    
    相比传统 LayerNorm 的优势:
      1. 彻底砍掉减去均值 (x - mu) 的步骤，反向传播少了一半计算链
      2. 训练极其稳定，LLaMA、Mistral、DeepSeek 全体标配
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在特征维度计算均方根
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return x_normed * self.weight


# =================================================================================
# 架构组件 2: 现代大模型标配 SwiGLU 门控前馈网络
# =================================================================================
class SwiGLU(nn.Module):
    """
    【架构技术细节: SwiGLU 门控前馈网络 (LLaMA / DeepSeek 核心)】
    
    数学公式:
        SwiGLU(x) = ( SiLU(x @ W_gate) * (x @ W_up) ) @ W_down
    
    技术原理:
      1. Gate 矩阵扮演“阀门”，输出动态平滑的开关信号
      2. Up 矩阵提取高维语义
      3. 两者逐位相乘 (*)，瞬间阻断无关噪声，只放行核心特征
      4. 工业界规范: 维度取 2/3 * 4 * d_model ≈ 2.67 倍，保持总参数量与传统 FFN 相当
    """
    def __init__(self, d_model: int, dropout: float = 0.15):
        super().__init__()
        hidden_dim = int(2 * 4 * d_model / 3)  # 约为 340 维，节约显存且保持公平参数量

        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up   = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu(x) 即 x * sigmoid(x)
        gate_out = F.silu(self.w_gate(x))
        up_out = self.w_up(x)
        return self.dropout(self.w_down(gate_out * up_out))


class ClassicFeedForward(nn.Module):
    """【传统 FFN 备选】: 2 矩阵 + GELU"""
    def __init__(self, d_model: int, dropout: float = 0.15):
        super().__init__()
        hidden_dim = 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BidirectionalMultiHeadAttentionV3(nn.Module):
    """双向多头注意力机制 (支持自定义头数与 Padding 屏蔽)"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.15):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.w_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask == 0, -1e9)

        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.w_o(out)


class TransformerBlockV3(nn.Module):
    """
    【V3 灵活编码器块】:
    支持按需开启 RMSNorm 与 SwiGLU
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_swiglu: bool = False,
        use_rmsnorm: bool = False,
        dropout: float = 0.15
    ):
        super().__init__()
        # 归一化层选型
        NormLayer = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.ln_1 = NormLayer(d_model)
        self.attn = BidirectionalMultiHeadAttentionV3(d_model, n_heads, dropout)

        self.ln_2 = NormLayer(d_model)
        # 前馈层选型
        self.mlp = SwiGLU(d_model, dropout) if use_swiglu else ClassicFeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), pad_mask=pad_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerClassifierV3(nn.Module):
    """
    【V3 消融实验主模型】:
    支持自由开关:
      - use_cls_token: 是否使用 BERT 经典的 [CLS] 专属决策标记
      - use_swiglu:    是否使用 LLaMA 的 SwiGLU 门控前馈网络
      - use_rmsnorm:   是否使用现代极简 RMSNorm
    """
    def __init__(
        self,
        vocab_size: int = 3500,
        num_classes: int = 15,
        d_model: int = 128,
        n_heads: int = 8,            # 升级为 8 头，捕捉更细粒度语境
        n_layers: int = 3,
        max_seq_len: int = 64,
        use_cls_token: bool = False, # 开关 1
        use_swiglu: bool = False,    # 开关 2
        use_rmsnorm: bool = False,   # 开关 3
        dropout: float = 0.15
    ):
        super().__init__()
        self.d_model = d_model
        self.use_cls_token = use_cls_token

        # 1. 词嵌入与位置编码
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        # 如果使用 [CLS]，序列长度需要多加 1 个位置
        actual_max_len = max_seq_len + 1 if use_cls_token else max_seq_len
        self.pos_emb = SinusoidalPositionalEncoding(d_model, actual_max_len)
        self.dropout = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # 核心细节: [CLS] Token 声明 (BERT 范式)
        # 一个单独的可学习向量，形状为 [1, 1, d_model]
        # ---------------------------------------------------------------------
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        else:
            # 不用 [CLS] 时，使用上一版的 Attention Pooling
            self.pooler = AttentionPooling(d_model)

        # 2. 堆叠 N 层 TransformerBlockV3
        self.blocks = nn.ModuleList([
            TransformerBlockV3(
                d_model=d_model,
                n_heads=n_heads,
                use_swiglu=use_swiglu,
                use_rmsnorm=use_rmsnorm,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # 3. 最终归一化
        NormLayer = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.ln_f = NormLayer(d_model)

        # 4. 分类预测头
        self.classifier = nn.Linear(d_model, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor):
        B, T = input_ids.shape

        # 1. 词嵌入
        x = self.token_emb(input_ids) # [B, T, d_model]

        # 2. 如果开启 [CLS]，在序列最前方拼上 [CLS] 标记
        if self.use_cls_token:
            # 将 [1, 1, d_model] 复制扩充为 [B, 1, d_model]
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1) # 形状变为 [B, T + 1, d_model]

            # 对应的 Padding 掩码也要在最前放行一个 True (因为 [CLS] 永远有效，不可屏蔽)
            cls_mask = torch.ones((B, 1), dtype=torch.bool, device=input_ids.device)
            raw_mask = (input_ids != 0)
            valid_mask = torch.cat([cls_mask, raw_mask], dim=1) # [B, T + 1]
            pad_mask = valid_mask.unsqueeze(1).unsqueeze(2)     # [B, 1, 1, T + 1]
        else:
            pad_mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)

        # 3. 注入位置编码
        x = self.pos_emb(x)
        x = self.dropout(x)

        # 4. 穿过多层 Transformer 编码层
        for block in self.blocks:
            x = block(x, pad_mask=pad_mask)

        x = self.ln_f(x)

        # 5. 提取整句语义特征 (决胜时刻！)
        if self.use_cls_token:
            # ★ BERT 范式: 直接抓取第 0 个位置 ([CLS] 对应的向量)！
            sentence_vec = x[:, 0, :] # [B, d_model]
        else:
            # 注意力池化
            sentence_vec, _ = self.pooler(x, pad_mask=pad_mask)

        sentence_vec = self.dropout(sentence_vec)
        logits = self.classifier(sentence_vec)
        return logits
