# src/init_rag.py

import logging
from src.rag_engine import rag_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_rag_system(force_reindex: bool = False):
    """
    Инициализирует RAG систему при старте приложения
    
    Args:
        force_reindex: Принудительно пересоздать индекс
    """
    try:
        logger.info("🚀 Инициализация RAG системы...")
        rag_agent.initialize_and_index(force_reindex=force_reindex)
        
        # Выводим статистику
        stats = rag_agent.get_stats()
        logger.info(f"📊 Статистика: {stats}")
        
        # Проверяем статус
        if stats['is_initialized']:
            logger.info("✅ RAG система успешно инициализирована")
            
            if stats.get('points_count', 0) > 0:
                logger.info(f"📊 Данные в Qdrant: {stats['points_count']} точек")
                
                # Тестовый запрос с обработкой ошибок
                try:
                    test_query = "Какие основные правила работы с поставщиками?"
                    logger.info(f"🔍 Тестовый запрос: {test_query}")
                    
                    # ИСПРАВЛЕНО: используем ask() вместо query()
                    test_response = rag_agent.ask(test_query)
                    
                    # Проверяем, что ответ не содержит ошибку
                    if test_response and "Ошибка" not in test_response:
                        logger.info(f"✅ Тестовый запрос выполнен: {test_response[:100]}...")
                    else:
                        logger.warning(f"⚠️ Тестовый запрос вернул: {test_response[:100]}...")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Тестовый запрос не выполнен: {e}")
            else:
                logger.warning("⚠️ Нет данных в Qdrant")
            
            return True
        else:
            logger.error("❌ RAG система не инициализирована")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации RAG: {e}", exc_info=True)
        return False

def get_rag_status():
    """Возвращает статус RAG системы"""
    try:
        return rag_agent.get_stats()
    except Exception as e:
        logger.error(f"❌ Ошибка получения статуса: {e}")
        return {"error": str(e)}