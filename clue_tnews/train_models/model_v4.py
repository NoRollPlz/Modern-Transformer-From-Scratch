"""
===================================================================================
项目名称: CLUE TNEWS 新闻分类竞赛实战
文件名称: model_v4.py
模型定位: V4 冠军完全体 (Champion Modern Transformer)
架构组合: RMSNorm (极简归一) + SwiGLU (门控前馈) + AttentionPooling (注意力加权池化)
===================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from train_models.model_v1 import TokenEmbedding, SinusoidalPositionalEncoding
except ImportError:
    from model_v1 import TokenEmbedding, SinusoidalPositionalEncoding


# =================================================================================
# 1. RMSNorm 均方根归一化 (LLaMA / Mistral / DeepSeek 标配)
# =================================================================================
class RMSNorm(nn.Module):
    """
    【架构技术细节: RMSNorm (Root Mean Square Normalization)】
    
    数学公式:
        RMS(x) = sqrt( mean(x^2) + eps )
        y = (x / RMS(x)) * weight
    
    核心优势:
        不减均值，只做均方根缩放，保留方向与幅度特征，反向传播效率更高。
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


# =================================================================================
# 2. SwiGLU 门控前馈网络 (现代大模型前馈层标杆)
# =================================================================================
class SwiGLU(nn.Module):
    """
    【架构技术细节: SwiGLU 门控前馈网络】
    
    数学公式:
        SwiGLU(x) = ( SiLU(x @ W_gate) * (x @ W_up) ) @ W_down
    
    核心优势:
        利用 Gate 作为“自适应阀门”，动态压制无用噪声，大幅提升关键特征置信度。
        隐藏层维度取 int(2 * 4 * d_model / 3)，在保持与传统 FFN 相同参数量的前提下提升表达力。
    """
    def __init__(self, d_model: int, dropout: float = 0.15):
        super().__init__()
        hidden_dim = int(2 * 4 * d_model / 3)  # 约为 340 维

        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up   = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_out = F.silu(self.w_gate(x))
        up_out = self.w_up(x)
        return self.dropout(self.w_down(gate_out * up_out))


# =================================================================================
# 3. 双向多头自注意力机制 (Bidirectional Multi-Head Attention)
# =================================================================================
class BidirectionalMultiHeadAttention(nn.Module):
    """
    双向多头注意力机制:
    在短文本分类中，允许字与字之间无障碍前后双向交互，并严格屏蔽 <PAD> 占位符。
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.15):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if pad_mask is not None:
            scores = scores.masked_fill(~pad_mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, T, C)

        return self.dropout(self.out_proj(context))


# =================================================================================
# 4. Modern Transformer 编码块 (Pre-RMSNorm + SwiGLU)
# =================================================================================
class ModernTransformerBlock(nn.Module):
    """
    【架构技术细节: Pre-RMSNorm 现代残差块】
    结构:
        x -> RMSNorm -> MultiHeadAttention -> 残差相加
          -> RMSNorm -> SwiGLU -> 残差相加
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.15):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = BidirectionalMultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLU(d_model, dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None):
        x = x + self.attn(self.norm1(x), pad_mask=pad_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# =================================================================================
# 5. Attention Pooling 注意力加权池化 (冠军提炼头)
# =================================================================================
class AttentionPooling(nn.Module):
    """
    【架构技术细节: Attention Pooling (注意力加权池化)】
    
    工作原理:
        1. 设立可学习全局主考官向量 query [d_model]
        2. 全文所有字向量与 query 计算注意力相关度打分
        3. 屏蔽 <PAD> 占位符后执行 Softmax 归一化
        4. 加权求和，精准抓取整句话的“提纲要领”
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None):
        # x: [B, T, d_model]
        # scores: [B, T]
        scores = torch.matmul(x, self.query) / math.sqrt(self.d_model)

        if pad_mask is not None:
            mask = pad_mask.squeeze(1).squeeze(1)  # [B, T]
            scores = scores.masked_fill(~mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)  # [B, T]
        sentence_vec = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # [B, d_model]
        return sentence_vec, attn_weights


# =================================================================================
# 6. V4 冠军分类模型完全体 (TransformerClassifierV4)
# =================================================================================
class TransformerClassifierV4(nn.Module):
    """
    【V4 完全体: RMSNorm + SwiGLU + AttentionPooling】
    
    流水线流程:
        Token IDs
           │
           ▼
        Token Embedding + Sinusoidal Positional Encoding
           │
           ▼
        N 层 ModernTransformerBlock (Pre-RMSNorm + Bidirectional MHA + SwiGLU)
           │
           ▼
        最终 RMSNorm 归一化
           │
           ▼
        Attention Pooling (聚光灯注意力加权池化)
           │
           ▼
        Dropout + Linear 分类头 (15 类别 Logits)
    """
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 15,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        max_seq_len: int = 64,
        dropout: float = 0.15
    ):
        super().__init__()
        self.d_model = d_model

        # 1. 词嵌入与位置编码
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb   = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.dropout   = nn.Dropout(dropout)

        # 2. 堆叠 N 层现代化 Transformer Block
        self.blocks = nn.ModuleList([
            ModernTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # 3. 最终归一化 (RMSNorm)
        self.ln_f = RMSNorm(d_model)

        # 4. 关键提炼头: Attention Pooling
        self.pooler = AttentionPooling(d_model)

        # 5. 分类输出头
        self.classifier = nn.Linear(d_model, num_classes)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor, return_attention: bool = False):
        # 1. 提取有效字符掩码 (PAD=0)
        pad_mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]

        # 2. 嵌入与位置编码
        x = self.token_emb(input_ids)
        x = self.pos_emb(x)
        x = self.dropout(x)

        # 3. 穿过多层现代化 Transformer Block
        for block in self.blocks:
            x = block(x, pad_mask=pad_mask)

        # 4. 最终归一化
        x = self.ln_f(x)

        # 5. 注意力加权池化提炼
        sentence_vec, attn_weights = self.pooler(x, pad_mask=pad_mask)

        # 6. 分类输出
        sentence_vec = self.dropout(sentence_vec)
        logits = self.classifier(sentence_vec)

        if return_attention:
            return logits, attn_weights
        return logits
