# src/agents/rag_agent.py

from typing import Dict, Any, Optional
import logging
from src.agents.base_agent import BaseAgent
from src.rag_engine import rag_agent

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):
    """Агент для работы с RAG системой и документами (гибридный поиск)"""
    
    def __init__(self):
        super().__init__(
            name="RAGAgent",
            description="Агент для поиска информации в документах компании с гибридным поиском (векторный + keyword + BM25 + графовый)",
            version="2.0.0"
        )
        self.rag = rag_agent
        self.logger.info("✅ RAGAgent инициализирован (гибридный поиск)")
    
    def process(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Поиск информации в документах с гибридным поиском"""
        self.logger.info(f"🔍 Гибридный поиск в документах: {query[:100]}...")
        
        try:
            # Инициализируем RAG если нужно
            if not self.rag.is_initialized:
                self.logger.info("Инициализация RAG системы...")
                self.rag.initialize_and_index()
            
            # Выполняем гибридный запрос
            result = self.rag.query(query)
            
            # Логируем использованные методы
            if result.get('hybrid_search', False):
                methods = result.get('methods_used', ['vector'])
                self.logger.info(f"📊 Использованы методы поиска: {', '.join(methods)}")
            
            return {
                "agent": self.name,
                "query": query,
                "result": result,
                "status": "success" if result.get("status") == "success" else "error",
                "sources": result.get("sources", []),
                "hybrid_search": result.get("hybrid_search", False),
                "methods_used": result.get("methods_used", ['vector']),
                "sources_count": result.get("sources_count", 0),
                "structured": result.get("structured")
            }
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка RAG: {e}", exc_info=True)
            return {
                "agent": self.name,
                "query": query,
                "result": {"answer": f"Ошибка при поиске в документах: {str(e)}"},
                "status": "error",
                "sources": [],
                "hybrid_search": False,
                "structured": None,
                "methods_used": []
            }
    
    def can_handle(self, query: str) -> bool:
        """Может обработать любые вопросы по документам"""
        return True
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Получить статистику по методам поиска"""
        stats = self.rag.get_stats() if self.rag else {}
        return {
            "hybrid_search_available": True,
            "methods": stats.get('methods', ['vector', 'keyword', 'bm25']),
            "use_graph": stats.get('use_graph', False),
            "graph_available": stats.get('graph_available', False),
            "is_initialized": stats.get('is_initialized', False),
            "chunks_count": stats.get('chunks_count', 0),
            "pages_available": stats.get('pages_available', [])
        }