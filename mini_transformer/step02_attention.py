"""
===================================================================================
项目名称: mini_transformer 从零实战
文件名称: 02_multi_head_attention.py
教学目标: 
    1. 纯手工实现缩放点积注意力 (Scaled Dot-Product Attention) 与因果掩码
    2. 深入实现多头注意力 (Multi-Head Attention) 的子空间投影、切分与拼接
    3. 数学验证实验: 亲手实测证明点积方差为什么是 d_k，以及除以 sqrt(d_k) 的必要性
    4. 观察非对称注意力矩阵如何打破“词与词对称性死锁”
===================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    【技术细节: 缩放点积注意力】
    数学公式:
        Attention(Q, K, V) = Softmax( (Q @ K.T) / sqrt(d_k) + Mask ) @ V
    """
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        输入维度:
            q: [batch_size, n_heads, seq_len_q, head_dim]
            k: [batch_size, n_heads, seq_len_k, head_dim]
            v: [batch_size, n_heads, seq_len_v, head_dim]
            mask: [1, 1, seq_len_q, seq_len_k] (下三角为1，右上角为0)
        """
        head_dim = q.size(-1)  # 即数学公式中的 d_k

        # ---------------------------------------------------------------------
        # 步骤 1: Q 与 K 的转置矩阵相乘 (最后两维相乘)
        # q: [..., seq_len_q, head_dim]
        # k.transpose(-2, -1): [..., head_dim, seq_len_k]
        # scores: [batch_size, n_heads, seq_len_q, seq_len_k]
        # ---------------------------------------------------------------------
        scores = torch.matmul(q, k.transpose(-2, -1))

        # ---------------------------------------------------------------------
        # 步骤 2: 缩放操作 (除以 sqrt(d_k))
        # 技术原因: 控制方差在 1 左右，防止进入 Softmax 梯度饱和区
        # ---------------------------------------------------------------------
        scores = scores / math.sqrt(head_dim)

        # ---------------------------------------------------------------------
        # 步骤 3: 因果掩码 (Causal Mask)
        # 将未来不可见的位置填入 -1e9，Softmax 后权重精确为 0
        # ---------------------------------------------------------------------
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # ---------------------------------------------------------------------
        # 步骤 4: Softmax 归一化为概率分布 (在最后一个维度上求和为 1.0)
        # ---------------------------------------------------------------------
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        # ---------------------------------------------------------------------
        # 步骤 5: 用注意力概率对 Value 进行加权求和
        # 乘积结果: [batch_size, n_heads, seq_len_q, head_dim]
        # ---------------------------------------------------------------------
        output = torch.matmul(attn_weights_dropped, v)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    【技术细节: 多头自注意力机制】
    将隐藏特征维度 d_model 拆分成 n_heads 个头，每个头负责 head_dim = d_model // n_heads。
    各个头在独立的几何子空间中并行计算注意力，最后拼接输出。
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除！"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # 线性投影层: 学习如何提取 Query, Key, Value
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # 核心注意力计算器
        self.attention = ScaledDotProductAttention(dropout=dropout)

        # 多头拼接后的最终输出整合层
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        batch_size = q.size(0)
        seq_len_q = q.size(1)
        seq_len_k = k.size(1)
        seq_len_v = v.size(1)

        # 1. 线性投影: [batch_size, seq_len, d_model]
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # 2. 拆分多头并转置:
        # [batch_size, seq_len, n_heads, head_dim] -> [batch_size, n_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_v, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. 计算注意力
        attn_output, attn_weights = self.attention(q, k, v, mask=mask)

        # 4. 拼接所有头的输出:
        # 转置回: [batch_size, seq_len_q, n_heads, head_dim]
        # view 回: [batch_size, seq_len_q, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        concat_output = attn_output.view(batch_size, seq_len_q, self.d_model)

        # 5. 最终输出映射
        return self.w_o(concat_output), attn_weights


def get_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """构造下三角因果掩码矩阵，形状为 [1, 1, seq_len, seq_len]"""
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
    return mask.unsqueeze(0).unsqueeze(0)


# =================================================================================
# 数学原理与技术细节验证实验
# =================================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" 阶段二：自注意力与多头注意力 深度技术细节实验")
    print("=" * 70)

    # -----------------------------------------------------------------------------
    # 实验 1: 亲手验证“除以根号 d_k”的数学方差推导
    # -----------------------------------------------------------------------------
    print("👉 [实验 1] 数学验证：点积的方差膨胀与 sqrt(d_k) 缩放效果")
    d_k_test = 64     # 假设头的维度是 64
    num_samples = 10000 # 生成一万对随机向量来统计方差

    # 生成均值为 0、方差为 1 的标准正态分布向量
    q_test = torch.randn(num_samples, d_k_test)
    k_test = torch.randn(num_samples, d_k_test)

    # 计算未缩放的点积: (q * k).sum(dim=-1)
    raw_dot_product = (q_test * k_test).sum(dim=-1)
    scaled_dot_product = raw_dot_product / math.sqrt(d_k_test)

    print(f"  测试维度 d_k = {d_k_test}")
    print(f"  - 未缩放时点积的理论方差应为 {d_k_test}，实际统计方差 = {raw_dot_product.var().item():.2f}")
    print(f"  - 缩放后点积的理论方差应为 1.0，实际统计方差   = {scaled_dot_product.var().item():.2f}")
    print("  💡 结论: 实验与数学推导完全吻合！除以 sqrt(d_k) 完美将暴增的方差拉回到了 1.0。")

    # -----------------------------------------------------------------------------
    # 实验 2: 多头注意力的形状与非对称权重流动检验
    # -----------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("👉 [实验 2] 多头因果自注意力的张量流转与非对称性检验")
    batch_size = 1
    seq_len = 4
    d_model = 16
    n_heads = 2
    head_dim = d_model // n_heads

    print(f"  超参数: batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}, n_heads={n_heads} (head_dim={head_dim})")

    # 构造假输入 (模拟经过 Embedding + Positional Encoding 后的张量)
    x = torch.randn(batch_size, seq_len, d_model)
    mask = get_causal_mask(seq_len)

    # 实例化并运行多头注意力层
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    out, attn_weights = mha(q=x, k=x, v=x, mask=mask)

    print(f"  输入 x 形状:         {list(x.shape)}")
    print(f"  输出 out 形状:       {list(out.shape)} (特征维度完全一致！)")
    print(f"  注意力权重矩阵形状:   {list(attn_weights.shape)} [batch, n_heads, seq_len, seq_len]")

    # 打印第 1 个注意力头的概率矩阵，观察非对称性与下三角
    print("\n  第 1 个头计算出的注意力概率矩阵 (精确到 3 位小数):")
    head_0_attn = attn_weights[0, 0].detach().numpy()
    for i in range(seq_len):
        row_str = [f"{val:.3f}" for val in head_0_attn[i]]
        print(f"    词 {i} 关注其他词 -> [{', '.join(row_str)}], 和 = {sum(head_0_attn[i]):.3f}")

    print("\n  💡 观察要点:")
    print("  1. 因果掩码起效: 右上角未来单词位置全部严格为 0.000。")
    print(f"  2. 非对称性成立: 词 2 对词 1 的关注度 ({head_0_attn[2, 1]:.3f}) 与 词 1 对词 2 的关注度 ({head_0_attn[1, 2]:.3f}) 完全不相等！")
    print("======================================================================")
