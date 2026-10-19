import os
import sys
import torch
import torch.nn as nn

# 添加当前目录
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import get_dataloaders
from train_models.model_v1 import TransformerClassifierV1
from train_models.model_v2 import TransformerClassifierV2
from train_models.model_v3 import TransformerClassifierV3
from train_models.model_v4 import TransformerClassifierV4

def evaluate_model(model, val_loader, criterion, device):
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
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, val_loader, tokenizer, label_names = get_dataloaders(batch_size=128, max_length=64)
    criterion = nn.CrossEntropyLoss()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "train_models")
    
    models_to_test = [
        {
            "id": "V1 Baseline",
            "desc": "Mean Pooling + Pre-LN + GELU FFN",
            "weights": "best_model_v1.pt",
            "init_fn": lambda: TransformerClassifierV1(
                vocab_size=tokenizer.vocab_size,
                num_classes=len(label_names),
                d_model=128, n_heads=4, n_layers=3, max_seq_len=64, dropout=0.1
            )
        },
        {
            "id": "V2 Attention",
            "desc": "AttentionPooling + 8Heads + Cosine LR",
            "weights": "best_model_v2.pt",
            "init_fn": lambda: TransformerClassifierV2(
                vocab_size=tokenizer.vocab_size,
                num_classes=len(label_names),
                d_model=128, n_heads=8, n_layers=3, max_seq_len=64, dropout=0.15
            )
        },
        {
            "id": "V3 [CLS]",
            "desc": "[CLS] Token 单测 (无 Pooling)",
            "weights": "best_model_v3_cls.pt",
            "init_fn": lambda: TransformerClassifierV3(
                vocab_size=tokenizer.vocab_size,
                num_classes=len(label_names),
                d_model=128, n_heads=8, n_layers=3, max_seq_len=64,
                use_cls_token=True, use_swiglu=False, use_rmsnorm=False, dropout=0.15
            )
        },
        {
            "id": "V3 SwiGLU",
            "desc": "SwiGLU 门控前馈单测 (Mean Pooling)",
            "weights": "best_model_v3_swiglu.pt",
            "init_fn": lambda: TransformerClassifierV3(
                vocab_size=tokenizer.vocab_size,
                num_classes=len(label_names),
                d_model=128, n_heads=8, n_layers=3, max_seq_len=64,
                use_cls_token=False, use_swiglu=True, use_rmsnorm=False, dropout=0.15
            )
        },
        {
            "id": "V3 RMSNorm",
            "desc": "RMSNorm 均方根归一化单测 (Mean Pooling)",
            "weights": "best_model_v3_rmsnorm.pt",
            "init_fn": lambda: TransformerClassifierV3(
                vocab_size=tokenizer.vocab_size,
                num_classes=len(label_names),
                d_model=128, n_heads=8, n_layers=3, max_seq_len=64,
                use_cls_token=False, use_swiglu=False, use_rmsnorm=True, dropout=0.15
            )
        },
        {
            "id": "V3 All In One",
            "desc": "[CLS] + SwiGLU + RMSNorm (瓶颈在CLS)",
            "weights": "best_model_v3_all.pt",
            "init_fn": lambda: TransformerClassifierV3(
                vocab_size=tokenizer.vocab_size,
                num_classes=len(label_names),
                d_model=128, n_heads=8, n_layers=3, max_seq_len=64,
                use_cls_token=True, use_swiglu=True, use_rmsnorm=True, dropout=0.15
            )
        },
        {
            "id": "V4 Champion",
            "desc": "RMSNorm + SwiGLU + AttentionPooling 冠军体",
            "weights": "best_model_v4_champion.pt",
            "init_fn": lambda: TransformerClassifierV4(
                vocab_size=tokenizer.vocab_size,
                num_classes=len(label_names),
                d_model=128, n_heads=8, n_layers=3, max_seq_len=64, dropout=0.15
            )
        }
    ]
    
    print("\n" + "=" * 90)
    print(f"{'模型代号':<16}{'架构核心特征':<38}{'参数量':<12}{'Val Loss':<12}{'Val Acc(%)':<12}")
    print("-" * 90)
    
    results = []
    test_sentences = [
        "歼20隐身战机震撼亮相珠海航展，双机编队上演空中芭蕾",
        "中央气象台发布暴雨黄色预警，华南江南局地有大暴雨",
        "英超天王山之战：阿森纳客场力克曼城重返积分榜首位",
        "华为发布全新折叠屏旗舰手机，自研鸿蒙系统流畅度大增"
    ]
    
    for item in models_to_test:
        path = os.path.join(models_dir, item["weights"])
        if not os.path.exists(path):
            print(f"⚠️ 跳过未找到的文件: {item['weights']}")
            continue
            
        model = item["init_fn"]().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        params = sum(p.numel() for p in model.parameters())
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        results.append({
            "id": item["id"],
            "desc": item["desc"],
            "params": params,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "model": model
        })
        print(f"{item['id']:<16}{item['desc']:<38}{params:<12,}{val_loss:<12.4f}{val_acc:<12.2f}")
        
    print("=" * 90)
    
    # 测试一句真实推理
    print("\n🎯 真实测试句验证对比 (以'歼20'战机为例):")
    for r in results:
        m = r["model"]
        m.eval()
        token_ids = tokenizer.encode(test_sentences[0], max_length=64)
        tokens = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            out = m(tokens)
            probs = torch.softmax(out, dim=-1)[0]
            pred_id = torch.argmax(probs).item()
            pred_name = label_names[pred_id]
            conf = probs[pred_id].item() * 100
        print(f"[{r['id']:<14}] 预测: {pred_name:<8} 置信度: {conf:.2f}%")

if __name__ == "__main__":
    main()
