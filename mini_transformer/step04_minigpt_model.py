"""
===================================================================================
项目名称: mini_transformer 从零实战
文件名称: step04_minigpt_model.py
教学目标: 
    1. 彻底串联所有积木: Embedding + Positional Encoding + N个 TransformerBlock + LM Head
    2. 掌握工业级显存优化技巧: 权重绑定 (Weight Tying)
    3. 掌握大模型训练关键细节: 权重初始化 (_init_weights, std=0.02) 与 Loss 校验
    4. 实现标准自回归交叉熵损失计算 (Cross-Entropy Loss)
    5. 实现完整的大模型打字机自回归生成循环 (.generate())
===================================================================================
"""

import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 保证当前目录模块能够正确导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from step01_embedding import TokenEmbedding, SinusoidalPositionalEncoding
from step02_attention import get_causal_mask
from step03_block import TransformerBlock


class MiniGPT(nn.Module):
    """
    【技术细节: 端到端 Decoder-Only 自回归语言模型整机】
    
    结构流向:
      1. 词嵌入与位置编码叠加: x = TokenEmb(idx) + PosEmb(pos)
      2. 穿过 N 个串联堆叠的 TransformerBlock (每一层都包含 MHA + FFN + Pre-LN + 残差)
      3. 最终层归一化 (Final LayerNorm)
      4. 语言模型投影头 (LM Head) 计算全词表得分 (Logits)
    """
    def __init__(
        self,
        vocab_size: int,       # 词表总大小 (如 1000)
        d_model: int = 64,     # 隐藏层特征维度
        n_heads: int = 4,      # 注意力头数
        n_layers: int = 4,     # 串联堆叠的 TransformerBlock 数量 (大楼层数)
        max_seq_len: int = 256,# 支持的最大上下文序列长度
        dropout: float = 0.0,
        tie_weights: bool = True # 是否开启权重绑定 (Weight Tying)
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # 1. 词嵌入与位置编码
        self.token_emb = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.pos_emb = SinusoidalPositionalEncoding(d_model=d_model, max_seq_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

        # -------------------------------------------------------------------------
        # 技术细节: nn.ModuleList
        # 必须使用 nn.ModuleList 存放多个 TransformerBlock，
        # PyTorch 才能自动追踪每一层内部的数十万个参数，并支持一键 .to(device)
        # -------------------------------------------------------------------------
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # 2. 最终层归一化 (Final LayerNorm)
        self.ln_f = nn.LayerNorm(d_model)

        # 3. 语言模型预测头 (LM Head): 将 d_model 映射为全词表的打分
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # -------------------------------------------------------------------------
        # 技术细节: 权重绑定 (Weight Tying)
        # 将输入 Embedding 的查找表权重与输出 LM Head 的权重共享为同一块物理显存。
        # 好处:
        # 1. 立刻节约 vocab_size * d_model 的参数量 (对小显存极其友好)
        # 2. 强迫模型用同一套几何流形去表达词向量的输入与输出，泛化性更强
        # -------------------------------------------------------------------------
        if tie_weights:
            self.lm_head.weight = self.token_emb.lut.weight

        # -------------------------------------------------------------------------
        # 技术细节: GPT 标准权重初始化 (_init_weights)
        # PyTorch 默认的 nn.Linear / Embedding 权重方差过大 (std=1.0)，
        # 在大模型中会导致初始点积数值极大，初期 Loss 飙升失真。
        # GPT 系列核心秘诀: 全部用 std=0.02 的正态分布初始化，确保初始盲猜概率均匀！
        # -------------------------------------------------------------------------
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        前向传播与训练损失计算
        
        输入:
            idx: 输入的 Token 编号序列，形状为 [batch_size, seq_len]
            targets: 目标真值序列 (右移一位的下一个词)，形状为 [batch_size, seq_len]，可选
        
        输出:
            logits: 未经归一化的词表打分，形状为 [batch_size, seq_len, vocab_size]
            loss: 交叉熵损失标量 (若提供 targets)
        """
        device = idx.device
        batch_size, seq_len = idx.shape
        assert seq_len <= self.max_seq_len, f"输入序列长度 {seq_len} 超过了最大支持长度 {self.max_seq_len}！"

        # 步骤 1: 构造防偷看的下三角因果掩码 [1, 1, seq_len, seq_len]
        mask = get_causal_mask(seq_len, device=device)

        # 步骤 2: 词向量与位置编码叠加
        x = self.token_emb(idx)
        x = self.pos_emb(x)
        x = self.dropout(x)

        # 步骤 3: 顺序穿过全部 N 层 TransformerBlock
        # 每一层都在残差总线上累加更深层的语义理解
        for block in self.blocks:
            x = block(x, mask=mask)

        # 步骤 4: 最终归一化与投影打分
        x = self.ln_f(x)
        logits = self.lm_head(x)  # 形状: [batch_size, seq_len, vocab_size]

        # 步骤 5: 如果提供了训练标签 targets，计算交叉熵损失
        loss = None
        if targets is not None:
            # F.cross_entropy 要求把前置批次维度打平为二维矩阵:
            # 预测值: [batch_size * seq_len, vocab_size]
            # 真实值: [batch_size * seq_len]
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None
    ) -> torch.Tensor:
        """
        【技术细节: 自回归文本生成循环 (打字机核心引擎)】
        
        原理:
          给模型一个前缀 Prompt，模型预测出下一个词的概率；
          用采样算法挑出下一个词，追加拼到原句子末尾；
          周而复始，像打字机一样一个词一个词往后吐字！
        """
        for _ in range(max_new_tokens):
            # 如果当前累计长度超过了 max_seq_len，截断保留最近的上下文窗口
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]

            # 1. 前向传播获取预测
            logits, _ = self(idx_cond)

            # 2. 我们只关心最后一个位置预测出的下一个词分布！
            # logits 原始形状: [batch_size, seq_len, vocab_size]
            # 取最后一维索引 -1 处: [batch_size, vocab_size]
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # 3. 可选操作: Top-k 截断采样 (避免模型胡言乱语)
            if top_k is not None:
                # 找出得分前 k 高的临界值，把低于前 k 的极端低分强行抹为负无穷
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # 4. Softmax 转化为和为 1.0 的概率分布
            probs = F.softmax(logits, dim=-1)

            # 5. 按照概率进行多项式轮盘赌采样 (具有一定的随机性和创造力)
            next_token = torch.multinomial(probs, num_samples=1)  # 形状: [batch_size, 1]

            # 6. 把新词拼接在已有句子末尾，作为下一轮的前缀输入！
            idx = torch.cat((idx, next_token), dim=1)

        return idx


# =================================================================================
# 实验验证与整机运行
# =================================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" 阶段四：从零手写组装完整 MiniGPT 整机模型")
    print("=" * 70)

    # 1. 自动选择硬件加速设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"👉 [1] 选用的计算硬件设备: {device}")

    # 2. 针对 6G 显存定制一个轻盈可控的 MiniGPT 超参数
    vocab_size = 1000      # 1000 个词
    d_model = 64           # 64 维特征
    n_heads = 4            # 4 个头
    n_layers = 4           # 4 层 TransformerBlock
    max_seq_len = 128      # 128 上下文

    print(f"👉 [2] 超参数配置: 词表={vocab_size}, 维度={d_model}, 头数={n_heads}, 层数={n_layers}")

    # 3. 实例化整机并迁移显存
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        tie_weights=True  # 启用权重绑定
    ).to(device)

    # 4. 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"👉 [3] 整机模型总参数量: {total_params:,} (约 {total_params / 1e3:.1f} K 参数，内存极其轻巧！)")

    # -----------------------------------------------------------------------------
    # 5. 模拟一次完整的训练前向传播 (计算 Loss)
    # -----------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("👉 [4] 模拟训练前向计算 (计算自回归交叉熵损失):")
    batch_size = 2
    cur_seq_len = 8
    
    # 模拟输入序列 x 与目标标签 y (y 是 x 往后错开一位的下一个词)
    dummy_x = torch.randint(0, vocab_size, (batch_size, cur_seq_len), device=device)
    dummy_y = torch.randint(0, vocab_size, (batch_size, cur_seq_len), device=device)

    logits, loss = model(dummy_x, targets=dummy_y)
    expected_loss = math.log(vocab_size)
    print(f"  输入 x 形状:          {list(dummy_x.shape)} [批大小, 序列长]")
    print(f"  预测 Logits 形状:     {list(logits.shape)} [批大小, 序列长, 全词表大小]")
    print(f"  实际计算出的初始 Loss: {loss.item():.4f}")
    print(f"  理论随机盲猜理想 Loss: ln({vocab_size}) = {expected_loss:.4f}")
    
    diff = abs(loss.item() - expected_loss)
    print(f"  💡 初始损失与理论值差异: {diff:.4f} (差异极小，证明权重初始化标准优雅！)")

    # -----------------------------------------------------------------------------
    # 6. 测试自回归打字机生成能力 (.generate())
    # -----------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("👉 [5] 触发打字机文本生成引擎 (.generate()):")
    
    # 给模型一个起始引导 Prompt (比如由 3 个词组成的句子前缀)
    prompt = torch.tensor([[12, 34, 56]], device=device)
    print(f"  输入的 Prompt 词 ID: {prompt.cpu().tolist()[0]}")

    model.eval()
    generated_sequence = model.generate(
        idx=prompt,
        max_new_tokens=10,  # 让它自回归吐出 10 个新词
        temperature=0.8,
        top_k=20
    )

    result_list = generated_sequence.cpu().tolist()[0]
    print(f"  自回归吐字生成的全序列: {result_list}")
    print(f"  成功从原有的 {prompt.size(1)} 个词，扩写生成到了 {len(result_list)} 个词！")
    print("=" * 70)
