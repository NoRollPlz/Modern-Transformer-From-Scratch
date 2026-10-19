"""
===================================================================================
项目名称: CLUE TNEWS 新闻分类竞赛实战
文件名称: train.py
模块功能: V1 Baseline 模型的完整训练、官方验证集精度评估与真机推理预测
===================================================================================
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import get_dataloaders, LABEL_NAMES
from train_models.model_v1 import TransformerClassifierV1


def evaluate(model, val_loader, criterion, device):
    """
    在 10,000 条官方验证集上进行准确率 (Accuracy) 与损失评估
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

    avg_loss = total_loss / total
    accuracy = (correct / total) * 100.0
    return avg_loss, accuracy


def train():
    print("=" * 75)
    print(" 🚀 CLUE TNEWS 竞赛实战：V1 Baseline 纯手写 Transformer 开训！")
    print("=" * 75)

    # 1. 设备检测
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"👉 [1] 选用的硬件加速设备: {device}")

    # 2. 超参数配置 (严格针对 6G 显存优化，追求最快训练吞吐)
    batch_size = 64        # 批大小
    max_length = 64        # 新闻标题最大长度 (平均 20~30 字)
    epochs = 4             # 训练轮次
    learning_rate = 1e-3   # 初始学习率
    d_model = 128          # 隐藏维度
    n_heads = 4            # 头数
    n_layers = 3           # 3 层 TransformerBlock
    dropout = 0.1

    print(f"👉 [2] 超参数设置: batch_size={batch_size}, lr={learning_rate}, d_model={d_model}, layers={n_layers}")

    # 3. 加载数据
    print("\n👉 [3] 正在加载数据流水线...")
    train_loader, val_loader, tokenizer, label_names = get_dataloaders(
        batch_size=batch_size,
        max_length=max_length
    )
    print(f"    训练集规模: {len(train_loader.dataset):,} 条 (共 {len(train_loader)} 个批次)")
    print(f"    验证集规模: {len(val_loader.dataset):,} 条 (官方客观评测基准)")

    # 4. 实例化 V1 Baseline 模型
    model = TransformerClassifierV1(
        vocab_size=tokenizer.vocab_size,
        num_classes=len(label_names),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_length,
        dropout=dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"👉 [4] V1 模型总参数量: {total_params:,} (约 {total_params/1e6:.2f} M 参数，显存占用极低！)")

    # 5. 优化器与交叉熵损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    # 6. 开始训练循环
    best_val_acc = 0.0
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "train_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model_v1.pt")

    print("\n" + "=" * 75)
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Train Acc (%)':<16}{'Val Loss':<14}{'Val Acc (%)':<14}{'耗时':<8}")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # 前向计算
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            # 反向求导与梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        # 本轮训练集指标
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        # 在官方 10,000 条验证集上评估
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        print(f"{epoch:<8}{epoch_train_loss:<14.4f}{epoch_train_acc:<16.2f}{val_loss:<14.4f}{val_acc:<14.2f}{elapsed:<8.1f}s")

        # 保存最佳模型权重
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print("-" * 75)
    print(f"🏆 V1 Baseline 最终官方验证集最高准确率: 【{best_val_acc:.2f}%】")
    print(f"📦 最佳权重已保存至: {save_path}")

    # -------------------------------------------------------------------------
    # 7. 实战推理：给模型输入 3 条新鲜真实新闻，看它怎么分类！
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print(" 🎯 实时新闻真机推理测试 (从验证集抽出真实样本):")
    print("=" * 75)
    
    test_samples = [
        "美联储宣布降息25个基点，全球资本市场迎来新变局",
        "歼20战机进行跨昼夜实战化飞行训练，空战能力大幅跃升",
        "英超天王山之战：阿森纳客场逆转曼城，登顶积分榜榜首",
        "全国各大高校考研初试分数线出炉，复试线出现明显分化",
    ]

    model.load_state_dict(torch.load(save_path))
    model.eval()

    with torch.no_grad():
        for text in test_samples:
            encoded_ids = tokenizer.encode(text, max_length=max_length)
            input_tensor = torch.tensor([encoded_ids], device=device)
            logits = model(input_tensor)
            pred_label = torch.argmax(logits, dim=-1).item()
            
            # 计算 Top-1 预测置信度
            prob = F.softmax(logits, dim=-1)[0, pred_label].item() * 100
            pred_name = label_names[pred_label]

            print(f"📰 新闻标题: 《{text}》")
            print(f"👉 AI 预测分类: 【{pred_name}】 (置信度: {prob:.1f}%)\n")

    print("=" * 75)


if __name__ == "__main__":
    train()
