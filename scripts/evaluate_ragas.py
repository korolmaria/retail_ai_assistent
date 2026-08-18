#!/usr/bin/env python3
"""
Скрипт для оценки качества RAG системы с помощью RAGAS
Использует локальную LM Studio для всех вычислений
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    context_precision, 
    context_recall
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.rag_engine import rag_agent
from src.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Класс для оценки RAG системы с помощью RAGAS (локальная LM Studio)"""
    
    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset_path = dataset_path or Path("data/datasets/test_dataset.json")
        self.results_dir = Path("data/datasets/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Настройка локальной LLM и эмбеддингов
        self._setup_local_llm()
    
    def _setup_local_llm(self):
        """Настройка LLM и эмбеддингов через LM Studio"""
        try:
            # Настройка LLM для оценки
            base_url = Config.LM_STUDIO_URL
            api_key = "not-needed"  # Для локальной LM Studio ключ не нужен
            model = Config.MODEL_NAME
            
            # Создаем LLM
            self.eval_llm = ChatOpenAI(
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=0.1,
                max_tokens=512,
            )
            self.ragas_llm = LangchainLLMWrapper(self.eval_llm)
            
            # Создаем эмбеддинги (используем ту же модель для эмбеддингов)
            # Для LM Studio эмбеддинги могут работать через тот же эндпоинт
            self.eval_embeddings = OpenAIEmbeddings(
                base_url=base_url,
                api_key=api_key,
                model=model,
            )
            self.ragas_embeddings = LangchainEmbeddingsWrapper(self.eval_embeddings)
            
            logger.info(f"✅ LLM и эмбеддинги настроены на {base_url}")
            logger.info(f"   Модель: {model}")
        except Exception as e:
            logger.error(f"❌ Ошибка настройки LLM: {e}")
            raise
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Загружает тестовый датасет"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"✅ Загружено {len(data)} вопросов из {self.dataset_path}")
            return data
        except FileNotFoundError:
            logger.error(f"❌ Файл датасета не найден: {self.dataset_path}")
            logger.info("💡 Создайте файл с вопросами или укажите правильный путь")
            raise
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки датасета: {e}")
            raise
    
    def prepare_ragas_dataset(self, test_data: List[Dict]) -> Dataset:
        """Подготавливает данные для RAGAS формата"""
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        logger.info("🔄 Получение ответов от RAG системы...")
        logger.info("=" * 60)
        
        for i, item in enumerate(test_data):
            question = item["question"]
            ground_truth = item.get("ground_truth", "")
            
            logger.info(f"  [{i+1}/{len(test_data)}] {question[:60]}...")
            
            # Получаем ответ от RAG
            result = rag_agent.query(question)
            
            answer = result.get("answer", "Нет ответа")
            answers.append(answer)
            
            # Извлекаем контексты (источники)
            contexts_list = []
            for src in result.get("sources", []):
                text = src.get("text", "")
                if text:
                    contexts_list.append(text[:500])  # Ограничиваем длину
            contexts.append(contexts_list)
            
            questions.append(question)
            ground_truths.append(ground_truth)
            
            logger.info(f"     ✅ Ответ получен, источников: {len(contexts_list)}")
        
        # Создаем датасет в формате RAGAS
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        
        logger.info(f"✅ Подготовлен датасет с {len(dataset)} примерами")
        return dataset
    
    def evaluate(self, dataset: Dataset) -> Dict[str, Any]:
        """Запускает оценку RAGAS с локальными моделями"""
        logger.info("🚀 Запуск RAGAS оценки (локальная LM Studio)...")
        logger.info("=" * 60)
        
        try:
            result = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall
                ],
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings,  # ← Добавляем эмбеддинги
            )
            
            logger.info("✅ Оценка завершена")
            return result
        except Exception as e:
            logger.error(f"❌ Ошибка оценки: {e}")
            raise
    
    def save_results(self, result: Dict[str, Any], dataset: Dataset):
        """Сохраняет результаты оценки"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохраняем результаты
        results_file = self.results_dir / f"results_{timestamp}.json"
        
        # Преобразуем результат в сериализуемый формат
        result_dict = {}
        for key, value in result.items():
            if hasattr(value, 'tolist'):
                result_dict[key] = value.tolist()
            elif hasattr(value, 'item'):
                result_dict[key] = value.item()
            else:
                result_dict[key] = value
        
        # Добавляем метаданные
        result_dict.update({
            "timestamp": timestamp,
            "num_samples": len(dataset),
            "dataset_path": str(self.dataset_path),
            "model": Config.MODEL_NAME,
            "llm_url": Config.LM_STUDIO_URL,
        })
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Результаты сохранены в {results_file}")
        
        # Сохраняем полный датасет с ответами
        dataset_file = self.results_dir / f"dataset_with_answers_{timestamp}.json"
        dataset_dict = dataset.to_dict()
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_dict, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Датасет с ответами сохранен в {dataset_file}")
        
        return results_file, dataset_file
    
    def print_results(self, result: Dict[str, Any]):
        """Выводит результаты в консоль"""
        print("\n" + "="*70)
        print("📊 РЕЗУЛЬТАТЫ RAGAS ОЦЕНКИ (локальная LM Studio)")
        print("="*70)
        
        metrics = {
            'faithfulness': 'Верность (отсутствие галлюцинаций)',
            'answer_relevancy': 'Релевантность ответа',
            'context_precision': 'Точность контекста',
            'context_recall': 'Полнота контекста'
        }
        
        print(f"\n🤖 Модель: {Config.MODEL_NAME}")
        print(f"📍 API: {Config.LM_STUDIO_URL}")
        print("\n📈 Метрики:")
        
        for key, name in metrics.items():
            if key in result:
                value = result[key]
                if hasattr(value, 'item'):
                    value = value.item()
                
                # Цветовая индикация
                if value >= 0.8:
                    status = "✅ Отлично"
                elif value >= 0.6:
                    status = "⚠️ Хорошо"
                elif value >= 0.4:
                    status = "🔶 Средне"
                else:
                    status = "❌ Требует улучшения"
                
                print(f"  {name}: {value:.4f}  {status}")
        
        print("\n" + "="*70)
        
        # Выводим интерпретацию
        print("\n💡 Интерпретация результатов:")
        
        if 'faithfulness' in result:
            f = result['faithfulness'].item() if hasattr(result['faithfulness'], 'item') else result['faithfulness']
            if f < 0.7:
                print("  • Верность низкая → LLM галлюцинирует. Улучшите промпт или модель.")
            else:
                print("  • Верность хорошая → LLM хорошо работает с контекстом.")
        
        if 'context_recall' in result:
            cr = result['context_recall'].item() if hasattr(result['context_recall'], 'item') else result['context_recall']
            if cr < 0.6:
                print("  • Полнота контекста низкая → Поиск пропускает важные документы. Увеличьте SEARCH_K.")
            else:
                print("  • Полнота контекста хорошая → Поиск находит нужные документы.")
        
        if 'context_precision' in result:
            cp = result['context_precision'].item() if hasattr(result['context_precision'], 'item') else result['context_precision']
            if cp < 0.6:
                print("  • Точность контекста низкая → Много шумных документов. Улучшите ретривер.")
            else:
                print("  • Точность контекста хорошая → Ретривер хорошо фильтрует.")
        
        if 'answer_relevancy' in result:
            ar = result['answer_relevancy'].item() if hasattr(result['answer_relevancy'], 'item') else result['answer_relevancy']
            if ar < 0.7:
                print("  • Релевантность ответа низкая → Ответ уходит от вопроса. Улучшите промпт.")
            else:
                print("  • Релевантность ответа хорошая → Ответ точно по вопросу.")
        
        print("="*70)
        
        # Выводим общий вердикт
        print("\n🎯 Общий вердикт:")
        avg_score = 0
        count = 0
        for key in metrics.keys():
            if key in result:
                value = result[key]
                if hasattr(value, 'item'):
                    value = value.item()
                avg_score += value
                count += 1
        
        if count > 0:
            avg_score /= count
            if avg_score >= 0.8:
                print("  ✅ Система работает отлично! Можно использовать в продакшене.")
            elif avg_score >= 0.6:
                print("  ⚠️ Система работает хорошо, но есть куда расти.")
            else:
                print("  ❌ Система требует улучшения. Рекомендуется доработка.")
        
        print("="*70)
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Запускает полный процесс оценки"""
        # 1. Загружаем датасет
        test_data = self.load_dataset()
        
        # 2. Подготавливаем данные для RAGAS
        dataset = self.prepare_ragas_dataset(test_data)
        
        # 3. Запускаем оценку
        result = self.evaluate(dataset)
        
        # 4. Сохраняем результаты
        self.save_results(result, dataset)
        
        # 5. Выводим результаты
        self.print_results(result)
        
        return {
            "result": result,
            "dataset": dataset,
            "num_samples": len(dataset)
        }


def main():
    """Главная функция"""
    print("🚀 ЗАПУСК RAGAS ОЦЕНКИ (локальная LM Studio)")
    print("=" * 70)
    
    # Проверяем, что RAG система инициализирована
    if not rag_agent.is_initialized:
        logger.info("🔄 Инициализация RAG системы...")
        rag_agent.initialize_and_index()
    
    # Проверяем наличие датасета
    dataset_path = Path("data/datasets/test_dataset.json")
    if not dataset_path.exists():
        print(f"\n❌ Датасет не найден: {dataset_path}")
        print("💡 Создайте файл data/datasets/test_dataset.json с вопросами")
        return
    
    # Проверяем LM Studio
    import requests
    try:
        response = requests.get(f"{Config.LM_STUDIO_URL}/models", timeout=5)
        if response.status_code == 200:
            models = response.json().get('data', [])
            print(f"✅ LM Studio доступна. Модели: {[m['id'] for m in models[:3]]}")
        else:
            print(f"⚠️ LM Studio отвечает с кодом {response.status_code}")
    except Exception as e:
        print(f"⚠️ Не удалось проверить LM Studio: {e}")
        print("💡 Убедитесь, что LM Studio запущена и сервер включен")
        response = input("\nПродолжить? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Создаем оценщик и запускаем
    evaluator = RAGEvaluator(dataset_path)
    results = evaluator.run_full_evaluation()
    
    print("\n✅ Оценка завершена успешно!")
    print(f"📊 Количество примеров: {results['num_samples']}")
    print(f"📁 Результаты сохранены в data/datasets/results/")


if __name__ == "__main__":
    main()