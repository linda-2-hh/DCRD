# -*- coding: utf-8 -*-
import os
import json
import glob
import random
import time
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.optim import AdamW
import torch
import wikipedia
from Bio import Entrez
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd
import csv

# ==================== 配置 =================
DATA_PATHS = {
    "strategyqa": "gpt/str_data",
    "openbookqa": "deepseek/obqa_data",
    "medqa": "GLM4.5_air/medqa_data"
}
LOCAL_MODEL_PATH = "google/flan-t5-large"
SAVE_MODEL_PATH = "model"
CHECKPOINT_PATH = "checkpoints"
TOP_K_DOCS = 3
ENTREZ_EMAIL = "****@gmail.com"
MAX_GLOBAL_EPOCHS = 2000
BASE_EPOCHS = 1
BATCH_SIZE = 4
MIN_GRAD_ACC = 4
MAX_GRAD_ACC = 12
LR = 3e-5
SAVE_EVERY_EPOCH = 2
EARLY_STOP_ACC = 0.90

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Entrez.email = ENTREZ_EMAIL

# ==================== 可视化 =================
def plot_loss_from_csv(log_dir="./loss_logs", save_dir="./plots"):
    os.makedirs(save_dir, exist_ok=True)
    all_loss_file = os.path.join(log_dir, "all_courses_loss.csv")
    if not os.path.exists(all_loss_file):
        print("⚠️ loss CSV 未找到，无法绘制 loss 曲线")
        return
    df = pd.read_csv(all_loss_file)
    plt.figure()
    for course in df["course"].unique():
        sub = df[df["course"] == course]
        plt.plot(sub["loss"].rolling(5).mean(), label=course)
    plt.title("Loss Curve by Course")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

def plot_course_intensity(learning_state, save_dir="./plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    for course, values in learning_state.history.items():
        plt.plot(values, label=course)
    plt.title("Course Learning Intensity Over Time (Pseudo Accuracy)")
    plt.xlabel("Training Steps")
    plt.ylabel("Pseudo Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "course_intensity.png"))
    plt.close()

def plot_weak_courses(learning_state, save_dir="./plots"):
    os.makedirs(save_dir, exist_ok=True)
    avg_scores = {c: sum(v)/len(v) if len(v)>0 else 0
                  for c, v in learning_state.history.items()}
    courses = list(avg_scores.keys())
    values = [1 - avg_scores[c] for c in courses]
    plt.figure()
    plt.bar(courses, values)
    plt.title("Weak Courses Ranking")
    plt.xlabel("Course")
    plt.ylabel("Weakness Score")
    plt.grid(axis="y")
    plt.savefig(os.path.join(save_dir, "weak_courses.png"))
    plt.close()

def draw_all_plots(learning_state, log_dir="./loss_logs", save_dir="./plots"):
    print("📊 开始绘制全部图表...")
    plot_loss_from_csv(log_dir, save_dir)
    plot_course_intensity(learning_state, save_dir)
    plot_weak_courses(learning_state, save_dir)
    print(f"📈 可视化完成，图像保存在：{save_dir}")

# ==================== Logger =================
class CourseLossLogger:
    def __init__(self, log_dir="./loss_logs2"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.history = {}

    def log(self, course_name, global_epoch, local_epoch, loss):
        self.history.setdefault(course_name, []).append(loss)
        course_file = os.path.join(self.log_dir, f"{course_name}_loss.csv")
        write_header = not os.path.exists(course_file)
        with open(course_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["global_epoch", "local_epoch", "loss"])
            writer.writerow([global_epoch, local_epoch, loss])

        global_file = os.path.join(self.log_dir, "all_courses_loss.csv")
        write_header_global = not os.path.exists(global_file)
        with open(global_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header_global:
                writer.writerow(["course", "global_epoch", "local_epoch", "loss"])
            writer.writerow([course_name, global_epoch, local_epoch, loss])

    # ===== 新增：获取最近 N 次 loss =====
    def get_recent_losses(self, course_name, n=20):
        hist = self.history.get(course_name, [])
        if len(hist) < n:
            return None
        return hist[-n:]

# ==================== Plateau 检测函数（新增） ====================
def check_plateau(loss_logger, courses, window=20, threshold=0.003):
    plateau_courses = []
    for name in courses:
        recent = loss_logger.get_recent_losses(name, n=window)
        if recent is None:
            return False  # 数据不足不能判断

        diffs = [abs(recent[i] - recent[i-1]) for i in range(1, len(recent))]
        avg_diff = sum(diffs) / len(diffs)

        if avg_diff <= threshold:
            plateau_courses.append(name)

    return len(plateau_courses) == len(courses)

# ==================== TeacherCourse =================
class TeacherCourse:
    def __init__(self, name, data_path):
        self.name = name
        self.files = sorted(glob.glob(f"{data_path}/*.json"))
        self.data = self._load_all()
        self.cursor = 0

    def _load_all(self):
        all_data = []
        for f in self.files:
            with open(f, "r", encoding="utf-8") as fp:
                all_data.append(json.load(fp))
        return all_data

    def get_lesson(self, batch_size=BATCH_SIZE):
        if self.cursor >= len(self.data):
            self.cursor = 0
            random.shuffle(self.data)
        batch = self.data[self.cursor:self.cursor + batch_size]
        self.cursor += batch_size
        return batch

# ==================== LearningState =================
class LearningState:
    def __init__(self):
        self.history = {}

    def update(self, course_name, acc):
        self.history.setdefault(course_name, []).append(acc)

    def recent_acc(self, course_name, n=5):
        hist = self.history.get(course_name, [])
        if len(hist) == 0:
            return 0.0
        return sum(hist[-n:]) / min(n, len(hist))

# ==================== KnowledgeRetriever =================
class KnowledgeRetriever:
    def __init__(self):
        wikipedia.set_lang("en")

    def retrieve_wikipedia(self, query, top_k=TOP_K_DOCS):
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

    def retrieve_pubmed(self, query, top_k=TOP_K_DOCS):
        docs = []
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=top_k)
            record = Entrez.read(handle)
            handle.close()
            for pmid in record["IdList"]:
                handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
                abstract = handle.read()
                handle.close()
                docs.append(abstract)
                time.sleep(0.1)
        except Exception as e:
            print(f"PubMed retrieval error: {e}")
        return docs

    def retrieve(self, course_name, query):
        if course_name == "medqa":
            return self.retrieve_pubmed(query)
        else:
            return self.retrieve_wikipedia(query)

# ==================== StudentModel =================
class StudentModel:
    def __init__(self, model_path=LOCAL_MODEL_PATH):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        if torch.cuda.device_count() > 1:
            print(f"🔹 使用 {torch.cuda.device_count()} GPUs 进行训练")
            self.model = torch.nn.DataParallel(self.model)
        self.model.train()
        self.optimizer = AdamW(self.model.parameters(), lr=LR)

    def train_on_batch(self, batch, grad_accum_steps):
        total_loss = 0.0
        self.optimizer.zero_grad()

        for i, item in enumerate(batch):
            input_text = f"""
You are a careful reasoning assistant. Use the following retrieved knowledge to answer the question.

Knowledge:
{item['retrieved_docs']}

Question: {item['question']}
Teacher Explanation: {item['teacher_explanation']}
Answer with the letter only (A/B/C/D):
"""
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
            labels = self.tokenizer(item['teacher_answer'], return_tensors="pt").input_ids.to(device)
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            total_loss += loss.item()

            if (i + 1) % grad_accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.optimizer.step()
        self.optimizer.zero_grad()
        return total_loss / len(batch)

    def save(self, path=SAVE_MODEL_PATH):
        os.makedirs(path, exist_ok=True)
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.save_pretrained(path)
        else:
            self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict() if not isinstance(self.model, torch.nn.DataParallel) else self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"🔹 已加载 checkpoint: {path}")
            return True
        return False

# ==================== CourseScheduler =================
class CourseScheduler:
    def __init__(self, courses, learning_state, min_acc=0.75, base_epochs=BASE_EPOCHS,
                 min_grad_acc=MIN_GRAD_ACC, max_grad_acc=MAX_GRAD_ACC):
        self.courses = courses
        self.state = learning_state
        self.min_acc = min_acc
        self.base_epochs = base_epochs
        self.min_grad_acc = min_grad_acc
        self.max_grad_acc = max_grad_acc

    def compute_course_weights(self):
        scores = {name: self.state.recent_acc(name) for name in self.courses}
        weights = {}
        for name, acc in scores.items():
            weights[name] = max(0.01, self.min_acc - acc)
        return weights

    def select_next_course(self):
        weights = self.compute_course_weights()
        total = sum(weights.values())
        probs = [weights[name]/total for name in self.courses]
        return random.choices(list(self.courses.keys()), weights=probs, k=1)[0]

    def get_dynamic_epoch_for_course(self, course_name):
        acc = self.state.recent_acc(course_name)
        dynamic_epoch = int(self.base_epochs + (self.min_acc - acc) * 5)
        return max(1, dynamic_epoch)

    def get_dynamic_grad_acc_for_course(self, course_name):
        acc = self.state.recent_acc(course_name)
        grad_acc = int(self.min_grad_acc + (self.min_acc - acc) * (self.max_grad_acc - self.min_grad_acc))
        return max(self.min_grad_acc, min(grad_acc, self.max_grad_acc))

# ==================== 主训练循环 =================
def train(student, scheduler, retriever, max_global_epochs=MAX_GLOBAL_EPOCHS, resume_checkpoint=True):
    writer = SummaryWriter(log_dir="t5/t5_large/kard/tensorboard_logs2")
    checkpoint_file = os.path.join(CHECKPOINT_PATH, "latest.pt")
    loss_logger = CourseLossLogger("t5/t5_large/kard/loss_logs2")

    start_epoch = 0
    if resume_checkpoint and student.load_checkpoint(checkpoint_file):
        start_epoch = int(checkpoint_file.split('_')[-1].replace(".pt","")) if "_" in checkpoint_file else 0

    for global_epoch in range(start_epoch, max_global_epochs):

        course_name = scheduler.select_next_course()
        course = scheduler.courses[course_name]
        dynamic_epoch = scheduler.get_dynamic_epoch_for_course(course_name)
        grad_accum_steps = scheduler.get_dynamic_grad_acc_for_course(course_name)

        for local_epoch in range(dynamic_epoch):
            raw_batch = course.get_lesson(batch_size=BATCH_SIZE)

            enhanced_batch = []
            for item in raw_batch:
                question = item.get("question", "")
                teacher_answer = item.get("answer", "")
                teacher_explanation = item.get("explanation", "")
                retrieved_docs = retriever.retrieve(course_name, question)
                retrieved_text = "\n".join(retrieved_docs[:TOP_K_DOCS])
                enhanced_batch.append({
                    "question": question,
                    "teacher_answer": teacher_answer,
                    "teacher_explanation": teacher_explanation,
                    "retrieved_docs": retrieved_text
                })

            loss = student.train_on_batch(enhanced_batch, grad_accum_steps)
            loss_logger.log(course_name, global_epoch, local_epoch, loss)
            writer.add_scalar(f"train/loss_{course_name}", loss, global_epoch*10 + local_epoch)
            print(f"[GlobalEpoch {global_epoch}, Course {course_name}, LocalEpoch {local_epoch}, GradAccum {grad_accum_steps}] loss={loss:.4f}")

            pseudo_acc = max(0, 1.0 - loss)
            scheduler.state.update(course_name, pseudo_acc)

        # ===== 保存 checkpoint =====
        if (global_epoch+1) % SAVE_EVERY_EPOCH == 0:
            checkpoint_file_epoch = os.path.join(CHECKPOINT_PATH, f"epoch_{global_epoch+1}.pt")
            student.save_checkpoint(checkpoint_file_epoch)
            latest_path = os.path.join(CHECKPOINT_PATH, "latest.pt")
            student.save_checkpoint(latest_path)
            student.save()
            print(f"💾 checkpoint 已保存: {checkpoint_file_epoch}")

        # ===== 原 pseudo-accuracy 早停 =====
        all_courses_acc = [scheduler.state.recent_acc(name) for name in scheduler.courses]
        if all(acc >= EARLY_STOP_ACC for acc in all_courses_acc):
            print(f"🎯 所有课程 pseudo accuracy >= {EARLY_STOP_ACC}，提前结束训练。")
            break

        # ===== Plateau-based Early Stopping（新增） =====
        if check_plateau(loss_logger, scheduler.courses.keys(), window=20, threshold=0.003):
            print("🎯 loss 变化进入 plateau，所有课程均已收敛，提前结束训练。")
            break

    writer.close()
    print(f"✅ 训练完成，最终模型已保存到 {SAVE_MODEL_PATH}")

    draw_all_plots(
        scheduler.state,
        log_dir="t5/t5_large/kard/loss_logs",
        save_dir="t5/t5_large/kard/plots"
    )

# ==================== 主程序 =================
if __name__ == "__main__":
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    courses = {name: TeacherCourse(name, path) for name, path in DATA_PATHS.items()}
    state = LearningState()
    scheduler = CourseScheduler(courses, state, min_acc=0.75, base_epochs=BASE_EPOCHS,
                                min_grad_acc=MIN_GRAD_ACC, max_grad_acc=MAX_GRAD_ACC)
    retriever = KnowledgeRetriever()
    student = StudentModel()
    train(student, scheduler, retriever)
