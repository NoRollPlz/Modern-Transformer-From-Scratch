"""
===================================================================================
项目名称: CLUE TNEWS 新闻分类竞赛实战
文件名称: train_v2.py
模块功能: V2 进阶模型训练 —— 带余弦退火学习率调度与注意力关键词热力可视化
===================================================================================
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import get_dataloaders, LABEL_NAMES
from train_models.model_v2 import TransformerClassifierV2


def evaluate(model, val_loader, criterion, device):
    """验证集精度评估"""
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

    return total_loss / total, (correct / total) * 100.0


def train():
    print("=" * 75)
    print(" 🚀 CLUE TNEWS 竞赛升级：V2 进阶版 (Attention Pooling + 余弦退火) 开训！")
    print(f" 📊 V1 历史最高基准战绩: 【50.66%】，今日目标: 突破并刷新纪录！")
    print("=" * 75)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"👉 [1] 计算硬件设备: {device}")

    # 超参数
    batch_size = 64
    max_length = 64
    epochs = 4
    init_lr = 1e-3
    d_model = 128
    n_heads = 4
    n_layers = 3
    dropout = 0.15  # 增强防过拟合正则化

    train_loader, val_loader, tokenizer, label_names = get_dataloaders(
        batch_size=batch_size,
        max_length=max_length
    )

    model = TransformerClassifierV2(
        vocab_size=tokenizer.vocab_size,
        num_classes=len(label_names),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_length,
        dropout=dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"👉 [2] V2 模型总参数量: {total_params:,} (仅比 V1 增加 128 个参数，保证绝对公平对比！)")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=1e-2)

    # -------------------------------------------------------------------------
    # 【技术细节: 余弦退火学习率调度器 (CosineAnnealingLR)】
    # 让学习率从 1e-3 按照平滑的余弦波浪曲线逐渐降至 1e-5，避免后期步子过大震荡
    # -------------------------------------------------------------------------
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "train_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model_v2.pt")

    print("\n" + "=" * 80)
    print(f"{'Epoch':<7}{'LR':<10}{'Train Loss':<13}{'Train Acc(%)':<14}{'Val Loss':<12}{'Val Acc(%)':<14}{'对比 V1':<10}")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        current_lr = optimizer.param_groups[0]["lr"]

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        # 步进学习率调度器
        scheduler.step()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        # 对比 V1 历史最高分 (50.66%)
        delta = val_acc - 50.66
        delta_str = f"+{delta:.2f}% 🟢" if delta > 0 else f"{delta:.2f}% 🔴"

        print(f"{epoch:<7}{current_lr:<10.1e}{epoch_train_loss:<13.4f}{epoch_train_acc:<14.2f}{val_loss:<12.4f}{val_acc:<14.2f}{delta_str:<10} ({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print("-" * 80)
    print(f"🏆 V2 最终官方验证集最高准确率: 【{best_val_acc:.2f}%】")
    print(f"📦 V2 最佳权重已保存至: {save_path}")

    # -------------------------------------------------------------------------
    # 7. 炫酷可视化：看 Attention Pooling 到底在关注新闻里的哪些关键词？
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 🎯 V2 真机推理与 Attention Pooling 关键词注意力透视:")
    print("=" * 80)
    
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
            
            # 同时获取预测分类与池化注意力权重
            logits, attn_weights = model(input_tensor, return_attn=True)
            pred_label = torch.argmax(logits, dim=-1).item()
            prob = F.softmax(logits, dim=-1)[0, pred_label].item() * 100
            pred_name = label_names[pred_label]

            # 找出注意力权重最高的前 3 个汉字
            weights = attn_weights[0].cpu().tolist()[:len(text)]
            top_char_indices = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)[:3]
            top_keywords = [f"'{text[i]}'(权重 {weights[i]*100:.1f}%)" for i in top_char_indices]

            print(f"📰 新闻标题: 《{text}》")
            print(f"👉 AI 预测分类: 【{pred_name}】 (置信度: {prob:.1f}%)")
            print(f"🔍 决策头最关注的 Top-3 关键词: {', '.join(top_keywords)}\n")

    print("=" * 80)


if __name__ == "__main__":
    train()
