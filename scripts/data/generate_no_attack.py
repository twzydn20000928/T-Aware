import json
import argparse
import os
from tqdm import tqdm

def main(args):
    if os.path.exists(args.output_file):
        raise FileExistsError(f"Refusing to overwrite {args.output_file}")

    print(f"📖 读取源文件: {args.input_file}")

    if not os.path.exists(args.input_file):
        print(f"❌ 错误: 文件不存在 {args.input_file}")
        return

    data = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except: continue

    print(f"📊 开始生成 No-Attack (Vanilla) 基准数据...")

    count_human = 0
    count_ai = 0

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for entry in tqdm(data, desc="Processing"):

            # --- 1. Human Data (Label=0) ---
            human_ans = entry.get('human_answer', '')
            # 兼容列表格式
            if not human_ans and 'human_answers' in entry and entry['human_answers']:
                 human_ans = entry['human_answers'][0]

            if human_ans:
                f.write(json.dumps({
                    "text": human_ans.strip(),
                    "label": 0,
                    "type": "human_original",
                    "id": entry.get("id")
                }, ensure_ascii=False) + '\n')
                count_human += 1

            # --- 2. AI Data (Standard ChatGPT, Label=1) ---
            # 直接提取原始生成的 AI 文本，不做任何改写
            ai_text = entry.get('chatgpt_answer', '')
            # 兼容列表格式
            if not ai_text and 'chatgpt_answers' in entry and entry['chatgpt_answers']:
                ai_text = entry['chatgpt_answers'][0]

            if ai_text:
                f.write(json.dumps({
                    "text": ai_text.strip(),
                    "label": 1,
                    # 标记为 standard，表示这是未受攻击的标准样本
                    "type": "ai_standard",
                    "id": entry.get("id")
                }, ensure_ascii=False) + '\n')
                count_ai += 1

    print(f"\n✅ No-Attack 数据集生成完毕！")
    print(f"   - Human 样本数: {count_human}")
    print(f"   - AI (Standard) 样本数: {count_ai}")
    print(f"📂 结果保存至: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate No-Attack (Vanilla) Dataset for Baseline Comparison")

    parser.add_argument("--input_file", type=str, required=True, help="源文件路径 (e.g., hc3.test.clean.jsonl)")
    parser.add_argument("--output_file", type=str, required=True, help="输出文件路径 (e.g., dataset.no_attack.hc3.jsonl)")

    args = parser.parse_args()
    main(args)
