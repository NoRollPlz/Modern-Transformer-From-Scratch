"""
===================================================================================
项目名称: CLUE TNEWS 新闻分类竞赛实战
文件名称: train_v4.py
执行目标: 训练 V4 冠军完全体 (RMSNorm + SwiGLU + AttentionPooling)
===================================================================================
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn

# 添加当前目录
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import get_dataloaders
from train_models.model_v4 import TransformerClassifierV4


def evaluate(model, val_loader, criterion, device):
    """在官方 10,000 条验证集上评估验证损失和准确率"""
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


def main():
    parser = argparse.ArgumentParser(description="训练 V4 冠军完全体架构")
    parser.add_argument("--epochs", type=int, default=4, help="训练轮数 (默认 4 轮)")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始最大学习率")
    args = parser.parse_args()

    print("=" * 85)
    print(" 🚀 CLUE TNEWS 决胜之战: 启动 V4 冠军完全体训练！")
    print(" 📌 终极三合一架构: RMSNorm (极简归一) + SwiGLU (门控前馈) + AttentionPooling (注意力池化)")
    print(" 📊 冲刺目标: 击败 V2 记录 (51.46%) 与 V1 基准 (50.66%)")
    print("=" * 85)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"👉 [1] 选用的计算设备: {device}")

    # 超参数设置
    batch_size = 64
    max_length = 64
    epochs = args.epochs
    init_lr = args.lr
    d_model = 128
    n_heads = 8
    n_layers = 3
    dropout = 0.15

    # 1. 加载数据
    train_loader, val_loader, tokenizer, label_names = get_dataloaders(
        batch_size=batch_size,
        max_length=max_length
    )

    # 2. 实例化 V4 模型
    model = TransformerClassifierV4(
        vocab_size=tokenizer.vocab_size,
        num_classes=len(label_names),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_length,
        dropout=dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"👉 [2] 模型参数量: {total_params:,} (轻量高效)")

    # 3. 优化器与余弦退火学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "train_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model_v4_champion.pt")

    print("\n" + "=" * 85)
    print(f"{'Epoch':<7}{'LR':<10}{'Train Loss':<13}{'Train Acc(%)':<14}{'Val Loss':<12}{'Val Acc(%)':<14}{'对比 V2 (51.46%)':<16}")
    print("-" * 85)

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

        scheduler.step()
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        delta = val_acc - 51.46
        delta_str = f"+{delta:.2f}% 🟢 创纪录!" if delta > 0 else f"{delta:.2f}% 🔴"

        print(f"{epoch:<7}{current_lr:<10.1e}{epoch_train_loss:<13.4f}{epoch_train_acc:<14.2f}{val_loss:<12.4f}{val_acc:<14.2f}{delta_str:<16} ({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print("-" * 85)
    print(f"🏆 【V4 冠军完全体】 官方验证集最高准确率: 【{best_val_acc:.2f}%】")
    print(f"📦 最佳模型权重已保存至: {save_path}")

    # 4. 实时新闻真机推理与注意力聚光灯透视
    print("\n" + "=" * 85)
    print(" 🎯 实时新闻真机推理测试与注意力聚光灯权重分析:")
    print("=" * 85)

    test_headlines = [
        "歼20隐身战机震撼亮相珠海航展，双机编队上演空中芭蕾",
        "中央气象台发布暴雨黄色预警，华南江南局地有大暴雨",
        "英超天王山之战：阿森纳客场力克曼城重返积分榜首位",
        "华为发布全新折叠屏旗舰手机，自研鸿蒙系统流畅度大增"
    ]

    # 加载最佳模型
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    for idx, sentence in enumerate(test_headlines, 1):
        token_ids = tokenizer.encode(sentence, max_length=max_length)
        x = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, attn_weights = model(x, return_attention=True)
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = torch.argmax(probs).item()
            pred_name = label_names[pred_id]
            conf = probs[pred_id].item() * 100

        # 取出前 5 个注意力得分最高的字
        valid_len = len(sentence)
        weights_valid = attn_weights[0, :valid_len].cpu().numpy()
        top_k_indices = weights_valid.argsort()[-4:][::-1]
        top_tokens = "".join([sentence[i] for i in top_k_indices if i < valid_len])

        print(f"[{idx}] 标题: 《{sentence}》")
        print(f"    👉 预测类别: 【{pred_name}】 | 置信度: 【{conf:.2f}%】")
        print(f"    🔍 聚光灯聚焦核心字: 【{top_tokens}】 (注意力峰值)")
        print("-" * 85)


if __name__ == "__main__":
    main()
