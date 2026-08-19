#!/usr/bin/env python3
"""
Скачивание flan-t5-small в папку models/
"""

import os
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Создаем папку для модели
model_path = Path("models/flan-t5-small")
model_path.mkdir(parents=True, exist_ok=True)

print(f"📥 Скачивание google/flan-t5-small в {model_path}")
print("⏳ Это может занять 2-5 минут...")

try:
    # Скачиваем модель и токенизатор
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    
    # Сохраняем в папку проекта
    tokenizer.save_pretrained(str(model_path))
    model.save_pretrained(str(model_path))
    
    print(f"✅ Модель сохранена в {model_path}")
    print(f"📁 Размер: {sum(f.stat().st_size for f in model_path.rglob('*')) / 1024 / 1024:.2f} MB")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")