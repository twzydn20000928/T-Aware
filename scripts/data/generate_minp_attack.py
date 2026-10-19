import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os

def main(args):
    if os.path.exists(args.output_file):
        raise FileExistsError(f"Refusing to overwrite {args.output_file}")

    print(f"🔄 正在加载模型: {args.model_path}")
    print(f"⚙️ 攻击配置: Min-P={args.min_p}, Temp={args.temperature}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # 确保 pad_token 设置正确
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_8bit=args.load_8bit  # 通过参数控制是否开启量化
    )
    model.eval()

    # Llama-3 官方 Prompt 模板
    def format_prompt(question):
        return (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    data = []
    print(f"📖 读取源文件: {args.input_file}")

    if not os.path.exists(args.input_file):
        print(f"❌ 错误: 文件 {args.input_file} 不存在！")
        return

    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except:
                    continue

    print(f"📊 开始生成 Min-P 攻击样本...")

    count_generated = 0
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for entry in tqdm(data, desc=f"MinP-Attack (p={args.min_p})"):

            # --- 1. Human Data (直接复制，Label=0) ---
            human_ans = entry.get('human_answer', '')
            if not human_ans and 'human_answers' in entry and entry['human_answers']:
                 human_ans = entry['human_answers'][0]

            if human_ans:
                f.write(json.dumps({
                    "text": human_ans,
                    "label": 0,
                    "type": "human",
                    "id": entry.get("id")
                }, ensure_ascii=False) + '\n')

            # --- 2. AI Data (Min-P 攻击, Label=1) ---
            prompt_text = entry.get('prompt', '') or entry.get('question', '')
            if not prompt_text: continue

            # 编码输入
            input_ids = tokenizer(format_prompt(prompt_text), return_tensors="pt").input_ids.to(model.device)

            try:
                with torch.no_grad():
                    # 【核心生成】
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=args.temperature, # 使用参数控制温度
                        min_p=args.min_p,             # 使用参数控制 Min-P
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

                # 截取新生成的文本
                generated_ids = outputs[0][input_ids.shape[-1]:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

                if generated_text.strip():
                    f.write(json.dumps({
                        "text": generated_text.strip(),
                        "label": 1,
                        "type": f"ai_min_p_{args.min_p}", # 记录具体的参数
                        "id": entry.get("id")
                    }, ensure_ascii=False) + '\n')
                    count_generated += 1

            except Exception as e:
                print(f"⚠️ 生成失败 (ID: {entry.get('id')}): {e}")
                continue

    print(f"\n✅ Min-P 攻击完成！共生成 {count_generated} 条 AI 样本。")
    print(f"📂 文件已保存至: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Min-P Adversarial Examples using Llama-3")

    # 核心路径参数
    parser.add_argument("--input_file", type=str, required=True, help="源文件路径 (.jsonl)")
    parser.add_argument("--output_file", type=str, required=True, help="输出文件路径 (.jsonl)")

    # 模型参数
    parser.add_argument("--model_path", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="模型路径或 HuggingFace ID")
    parser.add_argument("--load_8bit", action="store_true", help="是否开启 8-bit 量化 (节省显存)")

    # 攻击超参数 (Hyperparameters)
    parser.add_argument("--min_p", type=float, default=0.1,
                        help="Min-P 阈值 (2024 SOTA 推荐 0.05 ~ 0.1)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="采样温度 (Min-P 通常配合高温度使用，默认 1.0)")

    args = parser.parse_args()
    main(args)
