# src/agents/rag_agent.py
from typing import Dict, Any, Optional
import logging
from src.agents.base_agent import BaseAgent
from src.rag_engine import rag_agent

logger = logging.getLogger(__name__)

class RAGAgent(BaseAgent):
    """Агент для работы с RAG системой и документами"""
    
    def __init__(self):
        super().__init__(
            name="RAGAgent",
            description="Агент для поиска информации в документах компании (политики, инструкции, регламенты)",
            version="1.0.0"
        )
        self.rag = rag_agent
    
    def process(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Поиск информации в документах"""
        self.logger.info(f"🔍 Поиск в документах: {query}")
        
        try:
            # Инициализируем RAG если нужно
            if not self.rag.is_initialized:
                self.logger.info("Инициализация RAG системы...")
                self.rag.initialize_and_index()
            
            # Выполняем запрос
            result = self.rag.query(query, verbose=True)
            
            return {
                "agent": self.name,
                "query": query,
                "result": result,
                "status": "success" if result.get("status") == "success" else "error",
                "sources": result.get("sources", [])
            }
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка RAG: {e}")
            return {
                "agent": self.name,
                "query": query,
                "result": {"answer": f"Ошибка при поиске в документах: {str(e)}"},
                "status": "error",
                "sources": []
            }
    
    def can_handle(self, query: str) -> bool:
        """Может обработать любые вопросы по документам"""
        # Пока что может обработать всё
        return True