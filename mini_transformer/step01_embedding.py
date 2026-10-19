"""
===================================================================================
项目名称: mini_transformer 从零实战
文件名称: 01_embedding_and_position.py
教学目标: 
    1. 掌握词嵌入 (Token Embedding) 的内部查找机制与尺度缩放原理
    2. 掌握原始 Transformer 正弦/余弦绝对位置编码 (Sinusoidal PE) 的数学实现
    3. 亲眼见证：通过实验对比，观察位置编码如何破解自注意力的“语序置换不变性”
===================================================================================
"""

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    【技术细节: 词嵌入层】
    将离散的整数 Token ID 映射为连续密集的 d_model 维向量。
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # nn.Embedding 是一个 [vocab_size, d_model] 的权重矩阵 W_e
        # 每行代表一个词的特征向量
        self.lut = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x 形状为 [batch_size, seq_len]，存放每个词的整数 ID
        输出: 形状为 [batch_size, seq_len, d_model]
        
        技术细节:
        原始 Transformer 论文第 3.4 节特别指出，词嵌入矩阵的输出要乘以 sqrt(d_model)。
        目的是让词嵌入的数值尺度放大，避免后续与位置编码直接相加时被位置信息冲淡。
        """
        return self.lut(x) * math.sqrt(self.d_model)


class SinusoidalPositionalEncoding(nn.Module):
    """
    【技术细节: 正弦/余弦位置编码 (Sinusoidal Positional Encoding)】
    
    数学公式:
        PE(pos, 2i)   = sin( pos / (10000 ^ (2i / d_model)) )
        PE(pos, 2i+1) = cos( pos / (10000 ^ (2i / d_model)) )
    
    特点:
        完全没有可学习参数，是基于数学公式预先算好的确定性矩阵。
    """
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        # 1. 创建一个形状为 [max_seq_len, d_model] 的全零张量，用来保存位置编码
        pe = torch.zeros(max_seq_len, d_model)

        # 2. 生成物理位置序列: [0, 1, 2, ..., max_seq_len - 1]
        # .unsqueeze(1) 将其变成二维列向量，形状为 [max_seq_len, 1]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # 3. 计算分母的衰减项: 10000 ^ (2i / d_model)
        # 利用对数和指数等价变换: exp( (2i / d_model) * (-log(10000)) )
        # 这样计算可以在数值上更平稳，避免大数溢出
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 4. 偶数维度填入 sin，奇数维度填入 cos
        # 0::2 表示从索引 0 开始，步长为 2 (即 0, 2, 4, 6...)
        # 1::2 表示从索引 1 开始，步长为 2 (即 1, 3, 5, 7...)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 5. 扩充维度为 [1, max_seq_len, d_model]，方便后续与任意批次 batch_size 广播相加
        pe = pe.unsqueeze(0)

        # ---------------------------------------------------------------------
        # 技术细节: register_buffer
        # pe 不是神经网络的可训练参数（不需要计算梯度），但它是模型状态的一部分，
        # 必须随模型一起存盘保存，并随 model.to(device) 一起迁移到显存中。
        # register_buffer 专门用于注册这种“非参数但属于模型状态的张量”。
        # ---------------------------------------------------------------------
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x 为词嵌入输出，形状为 [batch_size, seq_len, d_model]
        输出: 词特征 与 位置特征 直接按元素相加，形状保持不变
        """
        # x.size(1) 是当前输入句子的实际长度 seq_len
        # 从预先算好的 pe 表中截取前 seq_len 个位置相加
        return x + self.pe[:, :x.size(1)]


# =================================================================================
# 实验对比验证：位置编码到底解决了什么问题？
# =================================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" 阶段：词嵌入与位置编码 技术细节深度实验")
    print("=" * 70)

    # 设定词表与超参数
    vocab = {"我": 1, "打": 2, "你": 3}
    vocab_size = len(vocab) + 1  # 词表大小为 4 (0 号留给特殊占位)
    d_model = 4                 # 设为较小的 4 维，方便肉眼直观查看矩阵数值
    max_seq_len = 10

    print(f"👉 基础设定: 词表 = {vocab}, 特征维度 d_model = {d_model}")

    # 构造两句词序相反的句子:
    # 句子 A: "我 打 你" -> [1, 2, 3]
    # 句子 B: "你 打 我" -> [3, 2, 1]
    sent_a = torch.tensor([[1, 2, 3]])
    sent_b = torch.tensor([[3, 2, 1]])

    # 1. 纯词嵌入测试 (不加位置编码)
    token_embedder = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
    emb_a = token_embedder(sent_a)
    emb_b = token_embedder(sent_b)

    print("\n" + "-" * 50)
    print("👉 [实验 1] 纯词嵌入输出 (观察第 1 个词与第 3 个词):")
    print(f"句子 A ('我 打 你') 的词向量集合包含:\n  '我': {emb_a[0, 0].detach().numpy()}\n  '你': {emb_a[0, 2].detach().numpy()}")
    print(f"句子 B ('你 打 我') 的词向量集合包含:\n  '你': {emb_b[0, 0].detach().numpy()}\n  '我': {emb_b[0, 2].detach().numpy()}")
    print("💡 结论: 如果没有位置信息，模型看到两句话的向量完全一致（只是换了个位置），没有任何先后关系！")

    # 2. 引入正弦位置编码
    pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_seq_len=max_seq_len)
    
    print("\n" + "-" * 50)
    print("👉 [实验 2] 查看位置编码矩阵 PE 的前 3 个位置数值:")
    sample_pe = pos_encoder.pe[0, :3].detach().numpy()
    for pos_idx in range(3):
        print(f"  位置索引 {pos_idx} 的位置向量: {sample_pe[pos_idx]}")

    # 3. 词向量 与 位置向量 相加
    final_a = pos_encoder(emb_a)
    final_b = pos_encoder(emb_b)

    print("\n" + "-" * 50)
    print("👉 [实验 3] 注入位置编码后的最终特征向量:")
    print(f"句子 A 中的 '我' (位于位置 0): {final_a[0, 0].detach().numpy()}")
    print(f"句子 B 中的 '我' (位于位置 2): {final_b[0, 2].detach().numpy()}")
    
    diff = torch.norm(final_a[0, 0] - final_b[0, 2]).item()
    print(f"\n💡 核心验证结果:")
    print(f"  同一个词 '我'，由于处于句首 (位置 0) 和句尾 (位置 2)，最终特征产生了明显差异 (欧氏距离 = {diff:.4f})！")
    print("  Transformer 成功获得了区分语序的能力！")
    print("=" * 70)
