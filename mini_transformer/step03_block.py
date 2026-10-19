"""
===================================================================================
项目名称: mini_transformer 从零实战
文件名称: step03_block.py
教学目标: 
    1. 掌握前馈网络 (Feed-Forward / MLP) 的维度放大与非线性加工技术细节
    2. 掌握 Pre-LN (前置层归一化) 相比 Post-LN 的架构优势
    3. 组装标准 TransformerBlock (自注意力 + 前馈网络 + 残差连接 + LayerNorm)
    4. 实验验证: 亲眼验证残差连接的“梯度高速公路”效应
===================================================================================
"""

import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 保证无论在哪个工作目录下运行，都能正确导入同目录模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入上一节手写的纯多头因果自注意力模块
from step02_attention import MultiHeadAttention, get_causal_mask


class FeedForward(nn.Module):
    """
    【技术细节: 前馈神经网络 (FFN / MLP)】
    
    数学公式:
        FFN(x) = GELU(x @ W_1) @ W_2
    
    技术原因:
        1. 维度放大 4 倍 (d_model -> 4 * d_model): 扩充高维特征空间容量，增强记忆拟合能力
        2. GELU 激活函数: 平滑的非线性变换，避免 ReLU 在负数区的硬性截断
        3. 逐位置独立处理: 词与词之间无交互，让每个词独立消化整合
    """
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        hidden_dim = 4 * d_model  # 经典工业界标准: 4 倍隐藏维度

        # 升维全连接层: [d_model -> 4 * d_model]
        self.w_1 = nn.Linear(d_model, hidden_dim, bias=False)
        # 降维全连接层: [4 * d_model -> d_model]
        self.w_2 = nn.Linear(hidden_dim, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x 形状为 [batch_size, seq_len, d_model]
        输出: 形状保持为 [batch_size, seq_len, d_model]
        """
        # 1. 升维并通过 GELU 激活
        # 2. 降维回 d_model
        return self.dropout(self.w_2(F.gelu(self.w_1(x))))


class TransformerBlock(nn.Module):
    """
    【技术细节: 单个标准 Transformer 解码器块】
    
    架构选择: Pre-LN (前置层归一化)
        x = x + Attention( LayerNorm_1(x) )
        x = x + MLP( LayerNorm_2(x) )
    
    技术优势:
        残差主干无任何阻拦，梯度直通到底层，支持极深网络的稳健训练。
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        # 子层 1 前置归一化
        self.ln_1 = nn.LayerNorm(d_model)
        # 多头因果自注意力子层
        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)

        # 子层 2 前置归一化
        self.ln_2 = nn.LayerNorm(d_model)
        # 前馈网络子层
        self.mlp = FeedForward(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播流向:
        两次 [LayerNorm -> 子层计算 -> 残差相加]
        """
        # 1. 注意力子层与残差连接
        normed_x1 = self.ln_1(x)
        attn_out, _ = self.attn(q=normed_x1, k=normed_x1, v=normed_x1, mask=mask)
        x = x + attn_out  # ★ 残差相加 (梯度高速公路 1)

        # 2. 前馈网络子层与残差连接
        normed_x2 = self.ln_2(x)
        mlp_out = self.mlp(normed_x2)
        x = x + mlp_out   # ★ 残差相加 (梯度高速公路 2)

        return x


# =================================================================================
# 技术细节验证实验
# =================================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" 阶段三：Transformer Block 架构与残差梯度流转实验")
    print("=" * 70)

    # 1. 超参数设置
    batch_size = 2
    seq_len = 4
    d_model = 16
    n_heads = 2

    print(f"👉 基础配置: batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}, n_heads={n_heads}")

    # 2. 构造模拟输入 (需要追踪梯度，方便验证反向传播)
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    mask = get_causal_mask(seq_len)

    # 3. 实例化 TransformerBlock
    block = TransformerBlock(d_model=d_model, n_heads=n_heads)

    # 4. 执行前向传播
    out = block(x, mask=mask)

    print("\n" + "-" * 50)
    print("👉 [实验 1] 形状与维度一致性检验:")
    print(f"  输入 x 形状:    {list(x.shape)}")
    print(f"  输出 out 形状:  {list(out.shape)} (特征维度严格守恒，支持任意多层堆叠！)")

    # -----------------------------------------------------------------------------
    # 5. 反向传播实验：验证残差连接如何守护梯度
    # -----------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("👉 [实验 2] 梯度穿透实验 (验证残差高速公路):")
    
    # 假设最终 Loss 为输出的标量和
    loss = out.sum()
    loss.backward()

    # 检查输入 x 所获得的梯度
    grad_norm = x.grad.norm().item()
    print(f"  对输入 x 的梯度范数 (Gradient Norm): {grad_norm:.4f}")
    
    # 核心判断: 如果没有残差连接，多层前向传播后梯度容易接近 0
    # 残差项中的常数 1 保证了即使各层权重很小，梯度也始终能稳定传递
    print(f"  输入 x 梯度均值: {x.grad.mean().item():.4f} (稳定非零)")
    print("  💡 结论: 梯度成功穿透整个 TransformerBlock，Pre-LN 与残差结构稳健运行！")
    print("=" * 70)
