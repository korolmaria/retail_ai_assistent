# src/init_rag.py

import logging
from typing import Dict, Any  # ← ДОБАВИТЬ ЭТУ СТРОКУ
from src.rag_engine import rag_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_rag_system(force_reindex: bool = False) -> bool:
    """
    Инициализирует RAG систему с гибридным поиском
    
    Args:
        force_reindex: Принудительно пересоздать индекс
    """
    try:
        logger.info("🚀 Инициализация ГИБРИДНОЙ RAG системы...")
        rag_agent.initialize_and_index(force_reindex=force_reindex)
        
        # Проверяем статус
        stats = rag_agent.get_stats()
        logger.info(f"📊 Статистика: {stats}")
        
        # Проверяем гибридные возможности
        if stats.get('is_initialized'):
            logger.info("✅ RAG система успешно инициализирована")
            
            # Выводим информацию о методах поиска
            methods = stats.get('methods', ['vector', 'keyword', 'bm25'])
            if stats.get('use_graph', False):
                methods.append('graph')
            
            logger.info(f"🔍 Доступные методы поиска: {', '.join(methods)}")
            logger.info(f"📚 Количество чанков: {stats.get('chunks_count', 0)}")
            logger.info(f"📄 Доступные страницы: {len(stats.get('pages_available', []))}")
            
            # Тестовый запрос с гибридным поиском
            # try:
            #     test_query = "Какие основные правила работы с поставщиками?"
            #     logger.info(f"🔍 Тестовый гибридный запрос: {test_query}")
                
            #     result = rag_agent.query(test_query)
                
            #     if result and result.get('status') == 'success':
            #         logger.info(f"✅ Тестовый запрос выполнен")
                    
            #         if result.get('hybrid_search', False):
            #             methods_used = result.get('methods_used', ['vector'])
            #             logger.info(f"📊 Использованы методы: {', '.join(methods_used)}")
                    
            #         sources_count = result.get('sources_count', 0)
            #         logger.info(f"📚 Найдено источников: {sources_count}")
                    
            #         # Выводим первые 100 символов ответа
            #         answer = result.get('answer', '')
            #         logger.info(f"💬 Ответ: {answer[:100]}...")
            #     else:
            #         logger.warning(f"⚠️ Тестовый запрос не выполнен")
                    
            # except Exception as e:
            #     logger.warning(f"⚠️ Тестовый запрос не выполнен: {e}")
            
            return True
        else:
            logger.error("❌ RAG система не инициализирована")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации RAG: {e}", exc_info=True)
        return False


def get_rag_status() -> Dict[str, Any]:
    """Возвращает статус RAG системы с информацией о гибридном поиске"""
    try:
        stats = rag_agent.get_stats()
        
        # Добавляем информацию о гибридном поиске
        stats['hybrid_search_enabled'] = True
        stats['methods'] = ['vector', 'keyword', 'bm25']
        if stats.get('use_graph', False):
            stats['methods'].append('graph')
        
        return stats
    except Exception as e:
        logger.error(f"❌ Ошибка получения статуса: {e}")
        return {"error": str(e)}