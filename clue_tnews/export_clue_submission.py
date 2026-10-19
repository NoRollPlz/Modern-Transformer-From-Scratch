"""
===================================================================================
项目名称: CLUE TNEWS 官方打榜提交生成器
文件名称: export_clue_submission.py
执行目标: 
  1. 读取 CLUE 官方 10,000 条未知真机测试集 (test.json)
  2. 使用 V4 冠军完全体 (best_model_v4_champion.pt) 进行批量高速推理
  3. 严格按照 CLUE 官方规范导出 tnews_predict.json
  4. 自动打包生成可直接提交的 tnews_submission.zip
===================================================================================
"""

import os
import sys
import json
import zipfile
import time
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# 添加当前目录
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import ChineseCharTokenizer
from train_models.model_v4 import TransformerClassifierV4

# CLUE 官方标准 15 分类代码与英文描述字典
LABEL_MAP = {
    0:  {"label": "100", "label_desc": "news_story",        "name": "故事/民生"},
    1:  {"label": "101", "label_desc": "news_culture",      "name": "文化/历史"},
    2:  {"label": "102", "label_desc": "news_entertainment","name": "娱乐/明星"},
    3:  {"label": "103", "label_desc": "news_sports",       "name": "体育/运动"},
    4:  {"label": "104", "label_desc": "news_finance",      "name": "财经/商业"},
    5:  {"label": "106", "label_desc": "news_house",        "name": "房产/楼市"},
    6:  {"label": "107", "label_desc": "news_car",          "name": "汽车/交通"},
    7:  {"label": "108", "label_desc": "news_edu",          "name": "教育/考试"},
    8:  {"label": "109", "label_desc": "news_tech",         "name": "科技/数码"},
    9:  {"label": "110", "label_desc": "news_military",     "name": "军事/军工"},
    10: {"label": "112", "label_desc": "news_travel",       "name": "旅游/出行"},
    11: {"label": "113", "label_desc": "news_world",        "name": "国际/时政"},
    12: {"label": "114", "label_desc": "news_stock",        "name": "证券/股票"},
    13: {"label": "115", "label_desc": "news_agriculture",  "name": "农业/农村"},
    14: {"label": "116", "label_desc": "news_game",         "name": "游戏/电竞"},
}


class TNewsTestDataset(Dataset):
    """用于快速批量推理的测试集加载器"""
    def __init__(self, raw_data, tokenizer, max_length=64):
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        item = self.raw_data[idx]
        tokens = self.tokenizer.encode(item["sentence"], max_length=self.max_length)
        return torch.tensor(tokens, dtype=torch.long), item["idx"], item["sentence"]


def main():
    print("=" * 80)
    print(" 🏁 启动 CLUE 官方测试集打榜推理流水线")
    print(" 📦 目标: 导出官方标准提交包 (tnews_predict.json & tnews_submission.zip)")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"👉 [1] 选用的推理硬件设备: {device}")

    # 1. 载入分词器
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(current_dir, "tnews_vocab.json")
    tokenizer = ChineseCharTokenizer(vocab_path)
    print(f"👉 [2] 成功加载专属分词字典，词表大小: {tokenizer.vocab_size}")

    # 2. 加载官方 10,000 条隐藏测试集
    print("👉 [3] 正在从 Hugging Face 载入 CLUE TNEWS 官方未知测试集 (test split)...")
    raw_test = load_dataset("clue/clue", "tnews", split="test")
    test_loader = DataLoader(
        TNewsTestDataset(raw_test, tokenizer, max_length=64),
        batch_size=128,
        shuffle=False
    )
    print(f"     ✅ 成功装载 {len(raw_test):,} 条官方盲测样本！")

    # 3. 加载 V4 冠军模型权重
    weights_path = os.path.join(current_dir, "train_models", "best_model_v4_champion.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"未找到 V4 冠军权重: {weights_path}，请先跑完 train_v4.py！")

    model = TransformerClassifierV4(
        vocab_size=tokenizer.vocab_size,
        num_classes=15,
        d_model=128,
        n_heads=8,
        n_layers=3,
        max_seq_len=64,
        dropout=0.15
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print("👉 [4] 成功装载 V4 冠军模型 (RMSNorm + SwiGLU + AttentionPooling)！")

    # 4. 开始批量极速推理
    print("\n👉 [5] 正在执行 10,000 条全量样本高速推理，请稍候...")
    start_time = time.time()
    submission_records = []
    preview_samples = []

    with torch.no_grad():
        for batch_tokens, batch_ids, batch_sentences in test_loader:
            batch_tokens = batch_tokens.to(device)
            logits = model(batch_tokens)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            batch_ids = batch_ids.tolist()

            for sample_id, pred_idx, sentence in zip(batch_ids, preds, batch_sentences):
                mapping = LABEL_MAP[pred_idx]
                record = {
                    "id": sample_id,
                    "label": mapping["label"],
                    "label_desc": mapping["label_desc"]
                }
                submission_records.append(record)

                if len(preview_samples) < 5:
                    preview_samples.append((sample_id, sentence, mapping["name"], mapping["label"]))

    elapsed = time.time() - start_time
    print(f"     ⚡ 10,000 条样本推理全部完成！耗时仅 {elapsed:.2f} 秒！({len(raw_test)/elapsed:.0f} 条/秒)")

    # 5. 写入官方格式 JSONL 文件
    output_dir = os.path.join(current_dir, "submission")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "tnews_predict.json")
    json_path_v11 = os.path.join(output_dir, "tnews1.1_predict.json")

    with open(json_path, "w", encoding="utf-8") as f:
        for r in submission_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 兼容 CLUE1.1 版本的命名
    with open(json_path_v11, "w", encoding="utf-8") as f:
        for r in submission_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n👉 [6] 官方标准预测文件已生成:")
    print(f"     📄 路径: {json_path} (共 {len(submission_records):,} 行)")

    # 6. 自动打包 zip (同时生成 tnews_submission.zip 和 submit.zip 确保官方裁判端识别)
    zip_paths = [
        os.path.join(output_dir, "tnews_submission.zip"),
        os.path.join(output_dir, "submit.zip")
    ]
    for zp in zip_paths:
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(json_path, arcname="tnews_predict.json")
            zipf.write(json_path_v11, arcname="tnews1.1_predict.json")

    print(f"👉 [7] 官方标准打榜提交压缩包已就绪:")
    print(f"     📦 压缩包路径 1: {zip_paths[0]}")
    print(f"     📦 压缩包路径 2: {zip_paths[1]} (官方推荐默认名)")

    # 7. 打印样例预览
    print("\n" + "=" * 80)
    print(" 🔍 官方盲测集真实预测样例预览 (前 5 条):")
    print("-" * 80)
    for sample_id, text, label_name, label_code in preview_samples:
        print(f"ID: {sample_id:<5} | 预测分类: 【{label_name}】({label_code}) | 标题: 《{text}》")
    print("=" * 80)

    print("\n 🎉 恭喜！你现在已具备正式提交打榜的全部材料！")


if __name__ == "__main__":
    main()
