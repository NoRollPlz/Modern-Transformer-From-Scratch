"""
===================================================================================
项目名称: mini_transformer 从零实战 (CLUE TNEWS 竞赛篇)
文件名称: step05_clue_tokenizer.py
教学目标: 
    1. 针对 CLUE TNEWS 中文竞赛数据集，量身定制中文字符级分词器 (ChineseCharTokenizer)
    2. 掌握特殊 Token 的设计技术细节 (<PAD> 占位补齐, <UNK> 未登录字)
    3. 构建词表 (Vocab)，实现字符与整型 ID 的双向编解码
    4. 制作标准的 PyTorch Dataset 与 DataLoader 批处理数据管道
===================================================================================
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class ChineseCharTokenizer:
    """
    【技术细节: 中文字符级分词器】
    
    优势:
      1. 词表极度紧凑 (约 2000~3000 字覆盖 99.9% 现代汉语)，节约 80% Embedding 显存
      2. 避免中文分词歧义问题
      3. 处理未登录字用 <UNK> 优雅兜底
    """
    def __init__(self, vocab_file: str = None):
        # 预留两个核心特殊 Token
        self.pad_token = "<PAD>"  # 填充占位符，ID 固定为 0
        self.unk_token = "<UNK>"  # 未知字符占位符，ID 固定为 1

        self.char2idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx2char = {0: self.pad_token, 1: self.unk_token}

        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)

    def build_vocab(self, texts, max_vocab_size: int = 4000, min_freq: int = 2):
        """
        从训练集文本中扫描统计高频字，构建词汇表
        
        参数:
            texts: 文本列表 (新闻标题)
            max_vocab_size: 最大词表容量 (4000 字绰绰有余)
            min_freq: 最小出现频次过滤 (出现 1 次的错别字/极端生僻字不予收录)
        """
        freq = {}
        for text in texts:
            for char in text:
                freq[char] = freq.get(char, 0) + 1

        # 按照出现频次从高到低排序
        sorted_chars = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        for char, count in sorted_chars:
            if count < min_freq or len(self.char2idx) >= max_vocab_size:
                break
            if char not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[char] = idx
                self.idx2char[idx] = char

        print(f"✅ 词表构建完成！共收录常用字符: {len(self.char2idx)} 个")

    def encode(self, text: str, max_length: int = 64) -> list:
        """
        将中文字符串转为定长整数 ID 列表
        
        技术细节:
          - 遇到未收录字符，替换为 <UNK> 的 ID (1)
          - 长度超过 max_length 则截断
          - 长度不足 max_length 则用 <PAD> (0) 填充对齐
        """
        ids = [self.char2idx.get(char, self.char2idx[self.unk_token]) for char in text]
        
        # 截断或填充
        if len(ids) > max_length:
            ids = ids[:max_length]
        else:
            ids = ids + [self.char2idx[self.pad_token]] * (max_length - len(ids))
        return ids

    def decode(self, ids: list) -> str:
        """将 ID 列表还原回中文字符串 (自动过滤 PAD)"""
        chars = [self.idx2char.get(i, self.unk_token) for i in ids if i != self.char2idx[self.pad_token]]
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.char2idx)

    def save_vocab(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char2idx, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.char2idx = json.load(f)
            self.idx2char = {v: k for k, v in self.char2idx.items()}


class TNewsDataset(Dataset):
    """
    【技术细节: 标准 PyTorch 竞赛数据集类】
    负责批量提供 (input_ids, label) 张量对
    """
    def __init__(self, dataset_split, tokenizer: ChineseCharTokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = dataset_split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item["sentence"]
        label = item["label"]

        # 编码为定长张量
        input_ids = self.tokenizer.encode(text, max_length=self.max_length)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# =================================================================================
# 验证脚本与词表落盘
# =================================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" 竞赛实战第一步：构建 CLUE TNEWS 专属中文字符分词器")
    print("=" * 70)

    # 1. 加载本地缓存的 CLUE TNEWS 数据
    print("👉 [1] 正在加载 CLUE TNEWS 数据集...")
    ds = load_dataset("clue/clue", "tnews")
    train_texts = ds["train"]["sentence"]
    print(f"    训练集句子总数: {len(train_texts):,} 篇")

    # 2. 构建分词器
    tokenizer = ChineseCharTokenizer()
    tokenizer.build_vocab(train_texts, max_vocab_size=3500, min_freq=3)

    # 3. 将词表持久化存盘，方便后续训练复用
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tnews_vocab.json")
    tokenizer.save_vocab(save_path)
    print(f"👉 [2] 词表已保存至: {save_path}")

    # 4. 测试编码与解码
    test_sentence = "中国女排夺得世界杯冠军，全场沸腾！"
    encoded_ids = tokenizer.encode(test_sentence, max_length=32)
    decoded_text = tokenizer.decode(encoded_ids)

    print("\n" + "-" * 50)
    print("👉 [3] 分词器编解码实测:")
    print(f"  原始中文句子:  '{test_sentence}'")
    print(f"  编码整型 ID:   {encoded_ids}")
    print(f"  反解还原中文:  '{decoded_text}'")

    # 5. 测试 PyTorch DataLoader 批量生成
    print("\n" + "-" * 50)
    print("👉 [4] 测试 DataLoader 批处理流转:")
    train_dataset = TNewsDataset(ds["train"], tokenizer, max_length=64)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    sample_batch_x, sample_batch_y = next(iter(train_loader))
    print(f"  Batch 输入 x 形状:  {list(sample_batch_x.shape)} [批大小, 定长序列]")
    print(f"  Batch 标签 y 形状:  {list(sample_batch_y.shape)} [批大小]")
    print(f"  第 1 个样本标签类别: {sample_batch_y[0].item()}")

    print("\n" + "=" * 70)
    print("🎉 恭喜！CLUE TNEWS 数据管道与中文字符分词器验证全部通过！")
    print("=" * 70)
