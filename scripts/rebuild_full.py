# scripts/rebuild_full.py

"""
Полная переиндексация с Camelot и OCR
"""

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_engine import rag_agent
from src.config import Config
from src.document_parser import document_parser
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_camelot():
    """Проверка Camelot"""
    try:
        import camelot
        logger.info(f"✅ Camelot версия: {camelot.__version__}")
        return True
    except ImportError:
        logger.warning("⚠️ Camelot не установлен")
        return False

def rebuild():
    """Полная переиндексация"""
    
    logger.info("=" * 60)
    logger.info("🚀 ЗАПУСК ПОЛНОЙ ПЕРЕИНДЕКСАЦИИ")
    logger.info("=" * 60)
    
    # Проверка Camelot
    camelot_available = check_camelot()
    
    # Проверка документов
    docs_dir = Config.DOCUMENTS_DIR
    pdf_files = list(docs_dir.glob("*.pdf"))
    logger.info(f"📄 Найдено PDF файлов: {len(pdf_files)}")
    
    for pdf in pdf_files:
        logger.info(f"   - {pdf.name}")
    
    # Очистка кэша
    logger.info("🗑️ Очистка кэша...")
    
    # Удаляем cache
    if Config.CACHE_DIR.exists():
        shutil.rmtree(Config.CACHE_DIR)
        logger.info(f"   ✅ Удален: {Config.CACHE_DIR}")
    
    # Удаляем docstore
    if Config.DOCSTORE_DIR.exists():
        shutil.rmtree(Config.DOCSTORE_DIR)
        logger.info(f"   ✅ Удален: {Config.DOCSTORE_DIR}")
    
    # Удаляем storage_context если есть
    storage_dir = Config.CACHE_DIR / "storage_context"
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
        logger.info(f"   ✅ Удален: {storage_dir}")
    
    logger.info("✅ Кэш очищен")
    
    # Индексация
    logger.info("📚 Индексация документов...")
    try:
        rag_agent.initialize_and_index(force_reindex=True)
        logger.info("✅ Индексация завершена")
    except Exception as e:
        logger.error(f"❌ Ошибка индексации: {e}")
        return
    
    # Статистика
    stats = rag_agent.get_stats()
    logger.info("\n📊 СТАТИСТИКА:")
    logger.info(f"   - Инициализирован: {stats.get('is_initialized', False)}")
    logger.info(f"   - Векторный индекс: {'✅' if stats.get('vector_index_loaded', False) else '❌'}")
    logger.info(f"   - Keyword индекс: {'✅' if stats.get('keyword_index_loaded', False) else '❌'}")
    logger.info(f"   - Графовый индекс: {'✅' if stats.get('graph_index_loaded', False) else '❌'}")
    logger.info(f"   - Документов в docstore: {stats.get('documents_in_docstore', 0)}")
    logger.info(f"   - Точек в Qdrant: {stats.get('qdrant_points', 0)}")
    
    # Тест
    logger.info("\n🔍 ТЕСТОВЫЙ ЗАПРОС:")
    test_query = "Что указано в Приложении 2 Информация о приемке товара?"
    logger.info(f"📝 Вопрос: {test_query}")
    
    try:
        result = rag_agent.query(test_query)
        answer = result.get('answer', 'Нет ответа')
        sources = result.get('sources', [])
        
        logger.info(f"📄 Ответ: {answer}")
        logger.info(f"📊 Найдено источников: {len(sources)}")
        
        if sources:
            logger.info("\n📚 ИСТОЧНИКИ:")
            for i, source in enumerate(sources[:3], 1):
                text = source.get('text', '')[:200]
                score = source.get('score', 'N/A')
                logger.info(f"   {i}. [score: {score}] {text}...")
        
        if 'Приложение 2' in answer:
            logger.info("\n✅ УСПЕШНО! Приложение 2 найдено!")
        else:
            logger.warning("\n⚠️ Приложение 2 не найдено. Проверьте парсинг PDF.")
            
            # Дополнительная диагностика
            logger.info("\n🔍 ДИАГНОСТИКА:")
            # Проверяем, есть ли Приложение 2 в docstore
            docstore_docs = list(rag_agent.docstore.docs.values()) if hasattr(rag_agent, 'docstore') else []
            logger.info(f"   Документов в docstore: {len(docstore_docs)}")
            
            # Ищем в тексте документов
            found = False
            for doc in docstore_docs[:5]:
                if hasattr(doc, 'text') and 'Приложение 2' in doc.text:
                    logger.info(f"   ✅ Найдено 'Приложение 2' в документе")
                    found = True
                    break
            
            if not found:
                logger.warning("   ❌ 'Приложение 2' не найдено в docstore")
                
    except Exception as e:
        logger.error(f"❌ Ошибка тестового запроса: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Переиндексация завершена!")
    logger.info("=" * 60)

if __name__ == "__main__":
    rebuild()