"""
===================================================================================
项目名称: CLUE TNEWS 新闻分类竞赛实战
文件名称: dataset.py
模块功能: 中文字符级分词器与 PyTorch 竞赛数据集/数据加载器流水线
===================================================================================
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


# CLUE TNEWS 官方 15 个类别的中文含义映射字典
LABEL_NAMES = [
    "100_故事/民生",
    "101_文化/历史",
    "102_娱乐/明星",
    "103_体育/运动",
    "104_财经/商业",
    "106_房产/楼市",
    "107_汽车/交通",
    "108_教育/考试",
    "109_科技/数码",
    "110_军事/军工",
    "112_旅游/出行",
    "113_国际/时政",
    "114_证券/股票",
    "115_农业/农村",
    "116_游戏/电竞",
]


class ChineseCharTokenizer:
    """
    【技术细节: 专属中文字符级分词器】
    """
    def __init__(self, vocab_file: str = None):
        self.pad_token = "<PAD>"  # 填充字符 (ID: 0)
        self.unk_token = "<UNK>"  # 未知字符 (ID: 1)

        self.char2idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx2char = {0: self.pad_token, 1: self.unk_token}

        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)

    def build_vocab(self, texts, max_vocab_size: int = 4000, min_freq: int = 3):
        """扫描文本构建高频字词表"""
        freq = {}
        for text in texts:
            for char in text:
                freq[char] = freq.get(char, 0) + 1

        sorted_chars = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        for char, count in sorted_chars:
            if count < min_freq or len(self.char2idx) >= max_vocab_size:
                break
            if char not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[char] = idx
                self.idx2char[idx] = char

    def encode(self, text: str, max_length: int = 64) -> list:
        """中文字符串转为定长整数 ID 列表 (自动截断或 PAD 补齐)"""
        ids = [self.char2idx.get(char, self.char2idx[self.unk_token]) for char in text]
        if len(ids) > max_length:
            ids = ids[:max_length]
        else:
            ids = ids + [self.char2idx[self.pad_token]] * (max_length - len(ids))
        return ids

    def decode(self, ids: list) -> str:
        """ID 列表反解回中文"""
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
    """PyTorch 标准数据集容器"""
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
        input_ids = self.tokenizer.encode(text, max_length=self.max_length)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def get_dataloaders(batch_size: int = 64, max_length: int = 64):
    """
    一键获取训练集与验证集的 DataLoader 以及分词器对象
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(current_dir, "tnews_vocab.json")

    # 1. 加载或构建分词器
    tokenizer = ChineseCharTokenizer()
    if os.path.exists(vocab_path):
        tokenizer.load_vocab(vocab_path)
    else:
        ds_raw = load_dataset("clue/clue", "tnews")
        tokenizer.build_vocab(ds_raw["train"]["sentence"], max_vocab_size=3500)
        tokenizer.save_vocab(vocab_path)

    # 2. 加载数据集
    ds = load_dataset("clue/clue", "tnews")
    train_dataset = TNewsDataset(ds["train"], tokenizer, max_length=max_length)
    val_dataset = TNewsDataset(ds["validation"], tokenizer, max_length=max_length)

    # 3. 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tokenizer, LABEL_NAMES


if __name__ == "__main__":
    train_loader, val_loader, tokenizer, label_names = get_dataloaders(batch_size=4)
    print(f"✅ 数据管道就绪！词表容量: {tokenizer.vocab_size}, 训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")
    x, y = next(iter(train_loader))
    print(f"样例输入 x: {x.shape}, 样例标签 y: {y.tolist()} -> {[label_names[i] for i in y.tolist()]}")
