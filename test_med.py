# -*- coding: utf-8 -*-
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import wikipedia
from Bio import Entrez
import time

# ================= 配置 =================
test_file = "medqa_usmle_hf/test.json"
local_model_path = "model"
output_file = "test/our_med.json"
knowledge_base = "pubmed"  # "wikipedia" 或 "pubmed"
top_k_docs = 3  # 每个问题检索 top k 文档

# PubMed 配置
Entrez.email = "liushuangxue231@gmail.com"  # 替换为你的邮箱

# ================= 加载模型 =================
print("🔹 Loading model and tokenizer from local path...")
tokenizer = T5Tokenizer.from_pretrained(local_model_path)
model = T5ForConditionalGeneration.from_pretrained(local_model_path)
model = model.eval()

# ================= 读取测试数据 =================
print("🔹 Reading test data...")
test_data = []
with open(test_file, "r", encoding="utf-8") as f:
    for line in f:
        test_data.append(json.loads(line))

label_map = {0: "A", 1: "B", 2: "C", 3: "D"}

# ================= 检索函数 =================
def retrieve_wikipedia(query, top_k=3):
    """返回 top_k 个 Wikipedia 摘要文本"""
    wikipedia.set_lang("en")
    try:
        results = wikipedia.search(query, results=top_k)
        docs = []
        for title in results:
            try:
                page = wikipedia.page(title)
                docs.append(page.summary)
            except:
                continue
        return docs
    except:
        return []

def retrieve_pubmed(query, top_k=3):
    """返回 top_k 个 PubMed 文档摘要"""
    docs = []
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=top_k)
        record = Entrez.read(handle)
        handle.close()
        id_list = record["IdList"]
        for pmid in id_list:
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
            abstract = handle.read()
            handle.close()
            docs.append(abstract)
            time.sleep(0.1)  # 避免被 PubMed 限制请求频率
    except Exception as e:
        print(f"PubMed retrieval error: {e}")
    return docs

# ================= 推理函数 =================
def ask_model(item):
    q = item["sent1"]
    options = [item["ending0"], item["ending1"], item["ending2"], item["ending3"]]

    # ================= 检索知识 =================
    knowledge_texts = []
    if knowledge_base == "wikipedia":
        knowledge_texts = retrieve_wikipedia(q, top_k=top_k_docs)
    elif knowledge_base == "pubmed":
        knowledge_texts = retrieve_pubmed(q, top_k=top_k_docs)

    knowledge_prompt = "\n".join(knowledge_texts)

    # ================= 构建 prompt =================
    input_text = f"""You are a careful reasoning assistant. The following are multiple-choice questions about medical knowledge.
Use the following retrieved knowledge to help answer the question.
Knowledge:
{knowledge_prompt}

Question: {q}
Options:
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}

Please answer with only the letter of the correct option (A, B, C, or D).
Answer:"""

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=8)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().upper()
    for opt in ["A", "B", "C", "D"]:
        if opt in answer:
            return opt
    return "N/A"

# ================= 执行推理 =================
print("🔹 Starting inference...")
total, correct = 0, 0

with open(output_file, "w", encoding="utf-8") as f_out:
    for item in tqdm(test_data, desc="Running inference"):
        gold_num = item.get("label", None)
        gold_letter = label_map.get(gold_num, "N/A")
        pred = ask_model(item)
        is_correct = (pred == gold_letter)
        result = {"id": item["id"], "gold": gold_letter, "pred": pred, "correct": is_correct}
        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
        total += 1
        correct += int(is_correct)

accuracy = correct / total if total > 0 else 0
print(f"\n✅ 推理完成，结果已保存至: {output_file}")
print(f"📊 总样本数: {total}")
print(f"🎯 准确率: {accuracy:.2%}")
