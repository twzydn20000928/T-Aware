import json
import random
import argparse
import os
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# --- 自动下载 NLTK 资源 ---
# WordNet 是同义词典，averaged_perceptron_tagger 是词性标注器
RESOURCES = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger_eng', 'wordnet', 'omw-1.4', 'stopwords']
for res in RESOURCES:
    try:
        nltk.data.find(f'tokenizers/{res}')
    except LookupError:
        try:
            nltk.data.find(f'corpora/{res}')
        except LookupError:
            try:
                nltk.data.find(f'taggers/{res}')
            except LookupError:
                print(f"⬇️ 正在下载 NLTK 资源: {res}...")
                nltk.download(res, quiet=True)

class SynonymAttacker:
    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        print("🔧 同义词攻击器已就绪 (基于 WordNet)")

    def get_wordnet_pos(self, treebank_tag):
        """把 NLTK 的词性标签转换为 WordNet 的格式"""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def get_synonyms(self, word, pos):
        """从 WordNet 获取同义词"""
        synonyms = set()
        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ")
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        return list(synonyms)

    def attack(self, text, perturbation_rate=0.1):
        """
        核心攻击函数
        perturbation_rate: 替换比例 (0.1 表示尝试替换 10% 的词)
        """
        if not text or not text.strip():
            return text

        words = word_tokenize(text)
        tagged_words = pos_tag(words)

        new_words = []
        num_replaced = 0

        for word, tag in tagged_words:
            # 1. 如果是停用词或标点，直接保留
            if word.lower() in self.stop_words or not word.isalnum():
                new_words.append(word)
                continue

            # 2. 只有满足概率要求才尝试替换
            if random.random() > perturbation_rate:
                new_words.append(word)
                continue

            # 3. 获取词性并查找同义词
            wn_pos = self.get_wordnet_pos(tag)
            if wn_pos:
                synonyms = self.get_synonyms(word, wn_pos)
                if synonyms:
                    # 随机选一个同义词
                    synonym = random.choice(synonyms)
                    new_words.append(synonym)
                    num_replaced += 1
                else:
                    new_words.append(word)
            else:
                new_words.append(word)

        # 简单的还原空格 (NLTK tokenize 会把空格去掉，这里简单用空格拼接)
        # 这种拼接在标点符号前会多出空格，对于检测器训练影响不大，但为了美观可以用 TreebankWordDetokenizer
        from nltk.tokenize.treebank import TreebankWordDetokenizer
        return TreebankWordDetokenizer().detokenize(new_words)

def main(args):
    if os.path.exists(args.output_file):
        raise FileExistsError(f"Refusing to overwrite {args.output_file}")

    attacker = SynonymAttacker()

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

    print(f"📊 开始同义词替换攻击 (Rate={args.perturbation_rate})...")

    count_generated = 0
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for entry in tqdm(data, desc="Synonym-Attack"):

            # --- 1. Human Data (Label=0) ---
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

            # --- 2. AI Data (同义词攻击, Label=1) ---
            ai_text = entry.get('chatgpt_answer', '')
            if not ai_text and 'chatgpt_answers' in entry and entry['chatgpt_answers']:
                ai_text = entry['chatgpt_answers'][0]

            if ai_text:
                try:
                    # 执行攻击
                    attacked_text = attacker.attack(ai_text, perturbation_rate=args.perturbation_rate)

                    if attacked_text.strip():
                        f.write(json.dumps({
                            "text": attacked_text.strip(),
                            "origin_text": ai_text,
                            "label": 1,
                            "type": f"ai_synonym_r{args.perturbation_rate}",
                            "id": entry.get("id")
                        }, ensure_ascii=False) + '\n')
                        count_generated += 1

                except Exception as e:
                    print(f"Error processing entry: {e}")
                    continue

    print(f"\n✅ 同义词攻击完成！共生成 {count_generated} 条 AI 样本。")
    print(f"📂 结果保存至: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Synonym Replacement Attacks using WordNet")

    parser.add_argument("--input_file", type=str, required=True, help="源文件路径 (.jsonl)")
    parser.add_argument("--output_file", type=str, required=True, help="输出文件路径 (.jsonl)")

    # 关键参数：扰动率
    parser.add_argument("--perturbation_rate", type=float, default=0.3,
                        help="同义词替换比例 (默认 0.3，即尝试替换 30%% 的词)")

    args = parser.parse_args()
    main(args)
