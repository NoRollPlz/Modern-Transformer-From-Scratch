"""
===================================================================================
项目名称: CLUE TNEWS 新闻分类竞赛实战
文件名称: model_v2.py
模块功能: V2 进阶模型 —— 引入 Attention Pooling (注意力加权池化) 决策头
===================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from train_models.model_v1 import (
        TokenEmbedding,
        SinusoidalPositionalEncoding,
        TransformerEncoderBlock
    )
except ImportError:
    from model_v1 import (
        TokenEmbedding,
        SinusoidalPositionalEncoding,
        TransformerEncoderBlock
    )


class AttentionPooling(nn.Module):
    """
    【V2 核心创新模块: 注意力加权池化 (Attention Pooling)】
    
    技术原理:
      1. 摒弃 V1 均值池化的死板平均，引入一个可学习的全局主考官向量 query [d_model]
      2. 每个字去和主考官算点积，决定自己的“关键词权重”
      3. 核心关键词 (如 "歼20", "考研", "降息") 获得极高权重，无意义虚词权重近乎为 0
      4. 严格屏蔽 <PAD> 占位符
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # 可学习的主考官向量 query，初始化为标准正态分布
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None):
        """
        输入:
            x: [batch_size, seq_len, d_model] 经过 Transformer 提炼的字向量
            pad_mask: [batch_size, 1, 1, seq_len]
        
        输出:
            sentence_vec: [batch_size, d_model] 加权提炼后的整句核心语义指纹
            attn_weights: [batch_size, seq_len] 每个字的注意力权重 (供可视化分析)
        """
        # 1. 计算每个字与主考官的点积得分
        # [batch_size, seq_len, d_model] @ [d_model] -> [batch_size, seq_len]
        scores = torch.matmul(x, self.query) / math.sqrt(self.d_model)

        # 2. 屏蔽 PAD 占位字符
        if pad_mask is not None:
            mask = pad_mask.squeeze(1).squeeze(1) # [batch_size, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)

        # 3. Softmax 计算每个汉字的归一化权重 (和为 1.0)
        attn_weights = F.softmax(scores, dim=-1) # [batch_size, seq_len]

        # 4. 按照注意力权重加权求和:
        # [batch_size, 1, seq_len] @ [batch_size, seq_len, d_model] -> [batch_size, 1, d_model]
        sentence_vec = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1) # [batch_size, d_model]

        return sentence_vec, attn_weights


class TransformerClassifierV2(nn.Module):
    """
    【V2 进阶模型整机】:
    双向 Transformer 编码器 + Attention Pooling 智能注意力决策头
    """
    def __init__(
        self,
        vocab_size: int = 3500,
        num_classes: int = 15,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        max_seq_len: int = 64,
        dropout: float = 0.15  # 适度调高 Dropout 抑制第 4 轮过拟合
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        # 1. 词嵌入与正弦位置编码
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        # 2. 堆叠 Transformer 编码层
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        # 3. ★ V2 核心升级: Attention Pooling 注意力加权池化层
        self.pooler = AttentionPooling(d_model)

        # 4. 最终分类预测线性头
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
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor, return_attn: bool = False):
        """
        前向传播
        """
        pad_mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)

        # 词嵌入 + 位置
        x = self.token_emb(input_ids)
        x = self.pos_emb(x)
        x = self.dropout(x)

        # 穿过所有 TransformerBlock
        for block in self.blocks:
            x = block(x, pad_mask=pad_mask)

        x = self.ln_f(x)

        # ★ 使用注意力池化提炼句子向量
        sentence_vec, attn_weights = self.pooler(x, pad_mask=pad_mask)
        sentence_vec = self.dropout(sentence_vec)

        # 分类打分
        logits = self.classifier(sentence_vec)

        if return_attn:
            return logits, attn_weights
        return logits


if __name__ == "__main__":
    model = TransformerClassifierV2()
    print(f"✅ V2 模型构建成功！总参数量: {sum(p.numel() for p in model.parameters()):,}")
    dummy_input = torch.randint(0, 3500, (2, 32))
    logits, attn = model(dummy_input, return_attn=True)
    print(f"Logits 形状: {list(logits.shape)}, 注意力权重形状: {list(attn.shape)}")
