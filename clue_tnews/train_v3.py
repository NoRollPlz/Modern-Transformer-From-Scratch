"""
===================================================================================
项目名称: CLUE TNEWS 新闻分类竞赛实战 (消融实验与架构演进篇)
文件名称: train_v3.py
模块功能: V3 多模式消融实验主控程序 (支持命令行自由指定测试单个架构或全要素合体)
===================================================================================
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import get_dataloaders, LABEL_NAMES
from train_models.model_v3 import TransformerClassifierV3


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


def main():
    parser = argparse.ArgumentParser(description="Transformer 核心架构消融实验")
    parser.add_argument(
        "--exp",
        type=str,
        default="all",
        choices=["cls", "swiglu", "rmsnorm", "all"],
        help="消融模式: cls (单测[CLS]标记), swiglu (单测门控前馈), rmsnorm (单测均方根归一), all (全要素合体)"
    )
    args = parser.parse_args()

    # 1. 根据模式匹配架构开关
    exp_configs = {
        "cls":     {"use_cls_token": True,  "use_swiglu": False, "use_rmsnorm": False, "name": "单测 [CLS] Token (BERT 经典决策范式)"},
        "swiglu":  {"use_cls_token": False, "use_swiglu": True,  "use_rmsnorm": False, "name": "单测 SwiGLU 门控前馈 (LLaMA / DeepSeek 核心)"},
        "rmsnorm": {"use_cls_token": False, "use_swiglu": False, "use_rmsnorm": True,  "name": "单测 RMSNorm 均方根归一化 (现代极简流)"},
        "all":     {"use_cls_token": True,  "use_swiglu": True,  "use_rmsnorm": True,  "name": "🔥 全要素合体完全体 ([CLS] + SwiGLU + RMSNorm)"},
    }

    cfg = exp_configs[args.exp]

    print("=" * 80)
    print(f" 🔬 CLUE TNEWS 消融实验开测！当前测试模式: 【{args.exp.upper()}】")
    print(f" 📌 架构特征: {cfg['name']}")
    print(f" 📊 对标历史基准: V1 Baseline = 50.66%, V2 AttentionPooling = 51.18%")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"👉 [1] 选用的计算设备: {device}")

    # 超参数
    batch_size = 64
    max_length = 64
    epochs = 4
    init_lr = 1e-3
    d_model = 128
    n_heads = 8  # 统一升级为 8 头，捕捉更细粒度语境
    n_layers = 3
    dropout = 0.15

    train_loader, val_loader, tokenizer, label_names = get_dataloaders(
        batch_size=batch_size,
        max_length=max_length
    )

    model = TransformerClassifierV3(
        vocab_size=tokenizer.vocab_size,
        num_classes=len(label_names),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_length,
        use_cls_token=cfg["use_cls_token"],
        use_swiglu=cfg["use_swiglu"],
        use_rmsnorm=cfg["use_rmsnorm"],
        dropout=dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"👉 [2] 模型参数量: {total_params:,} (轻巧可控)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "train_models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"best_model_v3_{args.exp}.pt")

    print("\n" + "=" * 80)
    print(f"{'Epoch':<7}{'LR':<10}{'Train Loss':<13}{'Train Acc(%)':<14}{'Val Loss':<12}{'Val Acc(%)':<14}{'对比 V2 (51.18%)':<16}")
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

        scheduler.step()
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        delta = val_acc - 51.18
        delta_str = f"+{delta:.2f}% 🟢" if delta > 0 else f"{delta:.2f}% 🔴"

        print(f"{epoch:<7}{current_lr:<10.1e}{epoch_train_loss:<13.4f}{epoch_train_acc:<14.2f}{val_loss:<12.4f}{val_acc:<14.2f}{delta_str:<16} ({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print("-" * 80)
    print(f"🏆 【{args.exp.upper()}】 官方验证集最高准确率: 【{best_val_acc:.2f}%】")
    print(f"📦 最佳权重保存至: {save_path}")

    # 4. 实时真机测试
    print("\n" + "=" * 80)
    print(" 🎯 实时新闻真机推理测试:")
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
            logits = model(input_tensor)
            pred_label = torch.argmax(logits, dim=-1).item()
            prob = F.softmax(logits, dim=-1)[0, pred_label].item() * 100
            print(f"📰 新闻标题: 《{text}》")
            print(f"👉 AI 预测: 【{label_names[pred_label]}】 (置信度: {prob:.1f}%)\n")


if __name__ == "__main__":
    main()
