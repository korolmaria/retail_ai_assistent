#!/usr/bin/env python3
"""
RAGAS ОЦЕНКА - с локальной Flan-T5
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENAI_API_KEY"] = "not-needed"

import json
import time
import torch
from datetime import datetime

from src.rag_engine import rag_agent
from src.config import Config

print("=" * 60)
print("🚀 RAGAS ОЦЕНКА (Flan-T5 локально)")
print("=" * 60)

# 1. Загружаем датасет
dataset_path = project_root / "data/datasets/test_dataset.json"
if not dataset_path.exists():
    print(f"❌ Файл не найден: {dataset_path}")
    sys.exit(1)

with open(dataset_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"✅ Загружено {len(test_data)} вопросов")

# 2. Получаем ответы от RAG
print("\n🔄 Получение ответов...")

questions = []
answers = []
contexts_list = []
ground_truths = []

for i, item in enumerate(test_data):
    question = item["question"]
    print(f"  [{i+1}/{len(test_data)}] {question[:40]}...")
    
    result = rag_agent.query(question)
    
    questions.append(question)
    answers.append(result.get("answer", ""))
    ground_truths.append(item.get("ground_truth", ""))
    
    contexts = []
    for src in result.get("sources", []):
        text = src.get("text", "")
        if text:
            contexts.append(text)
    contexts_list.append(contexts)

print(f"✅ Получено {len(questions)} ответов")

# 3. Импорты RAGAS
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from sentence_transformers import SentenceTransformer

# 4. Создаем датасет
dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts_list,
    "ground_truth": ground_truths,
})

# 5. Эмбеддинги - bge-m3
print("\n🔄 Загрузка эмбеддингов...")
embeddings_model = SentenceTransformer(
    str(Config.EMBEDDING_MODEL),
    device=Config.EMBEDDING_DEVICE,
    trust_remote_code=True,
)

class SimpleEmbeddings:
    def embed_documents(self, texts):
        return embeddings_model.encode(texts, normalize_embeddings=True).tolist()
    def embed_query(self, text):
        return embeddings_model.encode(text, normalize_embeddings=True).tolist()

embeddings = SimpleEmbeddings()
print("✅ Эмбеддинги загружены")

# ============================================================
# 6. ЛОКАЛЬНАЯ FLAN-T5 (НЕ LM Studio!)
# ============================================================
print("\n🔄 Настройка локальной Flan-T5...")

from transformers import T5Tokenizer, T5ForConditionalGeneration

class LocalFlanT5:
    def __init__(self):
        model_path = "models/flan-t5-small"
        print(f"   📥 Загрузка модели из: {model_path}")
        
        # Проверяем, что модель существует
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Модель не найдена в {model_path}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)
        print(f"   ✅ Модель загружена на {self.device}")
    
    def generate(self, messages, **kwargs):
        # Берем последнее сообщение как промпт
        prompt = messages[-1].get('content', '')
        
        # Токенизируем
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Генерируем
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=kwargs.get('max_new_tokens', 128),
            temperature=kwargs.get('temperature', 0.1),
            do_sample=True,
        )
        
        # Декодируем
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Создаем экземпляр
eval_llm = LocalFlanT5()
print("   ✅ LLM настроена (flan-t5-small)")

# 7. Запуск RAGAS
print("\n🚀 Запуск RAGAS...")
start = time.time()

try:
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=eval_llm,
        embeddings=embeddings,
    )
    elapsed = time.time() - start
    print(f"✅ Завершено за {elapsed:.2f} сек")
    
    # 8. Результаты
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ")
    print("=" * 60)
    
    metrics = {
        'faithfulness': 'Верность',
        'answer_relevancy': 'Релевантность',
        'context_precision': 'Точность контекста',
        'context_recall': 'Полнота контекста'
    }
    
    scores = {}
    for key, name in metrics.items():
        value = result[key]
        if isinstance(value, list):
            value = sum(value) / len(value) if value else 0
        if hasattr(value, 'item'):
            value = value.item()
        scores[key] = value
        status = "✅" if value > 0.7 else "⚠️" if value > 0.5 else "❌"
        print(f"  {name}: {value:.4f}  {status}")
    
    # 9. Сохраняем
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = project_root / "data/datasets/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    result_data = {
        "timestamp": timestamp,
        "num_samples": len(dataset),
        "metrics": scores,
        "model": "flan-t5-small",
        "embedding_model": str(Config.EMBEDDING_MODEL),
    }
    
    results_file = results_dir / f"ragas_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Результаты сохранены: {results_file}")

except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Готово!")