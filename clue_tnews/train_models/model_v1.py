"""
===================================================================================
项目名称: CLUE TNEWS 新闻分类竞赛实战
文件名称: model_v1.py
模块功能: V1 Baseline 基础 Transformer 中文分类模型架构
===================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    """【词嵌入层】: 映射字 ID 为密集向量"""
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.lut = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lut(x) * math.sqrt(self.d_model)


class SinusoidalPositionalEncoding(nn.Module):
    """【正弦位置编码】: 注入汉字在标题中的先后语序"""
    def __init__(self, d_model: int, max_seq_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class BidirectionalMultiHeadAttention(nn.Module):
    """
    【技术细节: 双向多头自注意力机制 (Bidirectional MHA)】
    
    分类任务关键特性:
      在新闻分类中，输入整句已知，无需遮挡未来词。
      汉字之间全向互联，但需要对补齐字符 <PAD> 做 Padding 掩码，防止无意义的空白干扰注意力。
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须被 n_heads 整除！"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        输入:
            x: [batch_size, seq_len, d_model]
            pad_mask: [batch_size, 1, 1, seq_len] 为 0 处表示 <PAD>，需要屏蔽
        """
        B, T, _ = x.shape

        # 1. 投影并切分多头: [B, n_heads, T, head_dim]
        q = self.w_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 2. 点积打分与根号缩放: [B, n_heads, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 3. 应用 Padding 掩码 (将 <PAD> 处填入 -1e9)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask == 0, -1e9)

        # 4. Softmax 概率归一化与加权求和
        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn_weights, v)  # [B, n_heads, T, head_dim]

        # 5. 拼接多头并输出投影: [B, T, d_model]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.w_o(out)


class FeedForward(nn.Module):
    """【前馈神经网络】: 4 倍升维 + GELU 平滑激活 + 降维压缩"""
    def __init__(self, d_model: int, dropout: float = 0.1):
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


class TransformerEncoderBlock(nn.Module):
    """【标准编码器块】: Pre-LN + 双向自注意力 + 残差 + FFN + 残差"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = BidirectionalMultiHeadAttention(d_model, n_heads, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-LN 结构与残差相加
        x = x + self.attn(self.ln_1(x), pad_mask=pad_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerClassifierV1(nn.Module):
    """
    【V1 Baseline 基线模型: 纯手写 Transformer 中文短文本分类器】
    
    处理全流程:
      1. 输入整型字序列 (Token IDs)
      2. 词嵌入 + 正弦位置编码
      3. 穿过多层双向 Transformer 编码层
      4. 最终归一化 Final LayerNorm
      5. 均值池化 (Mean Pooling): 忽略 PAD 字符，对一句话的所有真实字向量求平均
      6. 分类全连接头: 输出 15 类预测 Logits
    """
    def __init__(
        self,
        vocab_size: int = 3500,
        num_classes: int = 15,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        max_seq_len: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        # 堆叠 N 层编码器块
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        # 最终分类线性头: [d_model -> 15 个类别]
        self.classifier = nn.Linear(d_model, num_classes)

        # GPT 标准正态初始化权重 (std=0.02)
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

    def forward(self, input_ids: torch.Tensor):
        """
        前向传播计算分类得分
        输入: input_ids 形状为 [batch_size, seq_len]
        输出: logits 形状为 [batch_size, 15]
        """
        # 构造 Padding 掩码: 0 处为 False (PAD)，有效字为 True
        # 形状扩展为 [batch_size, 1, 1, seq_len]，支持广播到多头打分矩阵
        pad_mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)

        # 1. 词嵌入与位置编码
        x = self.token_emb(input_ids)
        x = self.pos_emb(x)
        x = self.dropout(x)

        # 2. 逐层前向传播
        for block in self.blocks:
            x = block(x, pad_mask=pad_mask)

        x = self.ln_f(x)  # [batch_size, seq_len, d_model]

        # ---------------------------------------------------------------------
        # 步骤 5: 均值池化 (Mean Pooling) 技术细节
        # 我们不能简单地对 seq_len 维度做 mean()，因为后面有很多填充的 <PAD>！
        # 正确做法: 用 pad_mask 把 PAD 处的特征清零，然后只除以该句子的“真实字数”
        # ---------------------------------------------------------------------
        # mask 形状: [batch_size, seq_len, 1]
        valid_mask = (input_ids != 0).unsqueeze(-1).float()
        sum_embeddings = torch.sum(x * valid_mask, dim=1) # [batch_size, d_model]
        valid_lens = torch.clamp(valid_mask.sum(dim=1), min=1e-9) # 真实有效字数
        sentence_vec = sum_embeddings / valid_lens        # [batch_size, d_model]

        # ---------------------------------------------------------------------
        # 步骤 6: 分类预测头打分
        # ---------------------------------------------------------------------
        logits = self.classifier(sentence_vec)  # [batch_size, 15]
        return logits


if __name__ == "__main__":
    # 快速自检
    model = TransformerClassifierV1(vocab_size=3500, num_classes=15, d_model=128)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"✅ V1 Baseline 模型构建成功！总参数量: {total_p:,} (约 {total_p/1e3:.1f} K，极度轻盈)")
    dummy_input = torch.randint(0, 3500, (4, 32))
    dummy_out = model(dummy_input)
    print(f"测试输入形状: {list(dummy_input.shape)} -> 输出分类 Logits 形状: {list(dummy_out.shape)}")
