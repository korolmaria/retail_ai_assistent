# src/agents/supervisor.py
from typing import Dict, Any, List, Optional
import logging
from src.agents.base_agent import BaseAgent
from src.agents.rag_agent import RAGAgent
from src.agents.api_agents import OrdersAgent, GoodsAgent, ReportsAgent, AnalyticsAgent
from src.client import llamaindex_llm

logger = logging.getLogger(__name__)

class Supervisor:
    """Супервизор для управления агентами"""
    
    def __init__(self, mode: str = "rag_first"):
        """
        Args:
            mode: Режим работы
                - "rag_first": всегда использовать RAG агента
                - "router": маршрутизировать запросы к подходящим агентам
                - "hybrid": гибридный режим (RAG + специализированные агенты)
        """
        self.mode = mode
        self.agents: List[BaseAgent] = []
        self.default_agent = None
        self.logger = logging.getLogger("supervisor")
        self.query_history = []
        
        # Регистрируем всех агентов
        self._register_all_agents()
        
        self.logger.info(f"🚀 Супервизор инициализирован в режиме: {mode}")
        self.logger.info(f"📋 Зарегистрировано агентов: {len(self.agents)}")
    
    def _register_all_agents(self):
        """Регистрирует всех агентов"""
        # Регистрируем RAG агента первым (по умолчанию)
        self.rag_agent = RAGAgent()
        self.register_agent(self.rag_agent)
        
        # Регистрируем API агентов с заглушками
        self.register_agent(OrdersAgent())
        self.register_agent(GoodsAgent())
        self.register_agent(ReportsAgent())
        self.register_agent(AnalyticsAgent())
        
        # Устанавливаем RAG как агент по умолчанию
        self.default_agent = self.rag_agent
    
    def register_agent(self, agent: BaseAgent):
        """Регистрирует нового агента"""
        self.agents.append(agent)
        self.logger.info(f"✅ Зарегистрирован агент: {agent.name} ({agent.__class__.__name__})")
    
    def _select_agent(self, query: str) -> BaseAgent:
        """
        Выбирает подходящего агента для запроса в зависимости от режима
        """
        self.logger.info(f"🔍 Выбор агента для запроса: {query[:50]}...")
        
        # Режим RAG-first - всегда используем RAG агента
        if self.mode == "rag_first":
            self.logger.info(f"🔄 Режим RAG-first: используем {self.default_agent.name}")
            return self.default_agent
        
        # Режим роутера - пытаемся найти подходящего агента
        if self.mode == "router":
            # Находим всех агентов, которые могут обработать запрос
            capable_agents = [a for a in self.agents if a.can_handle(query)]
            
            if capable_agents:
                # Если есть несколько - выбираем с помощью LLM
                if len(capable_agents) > 1:
                    selected = self._llm_select_agent(query, capable_agents)
                    self.logger.info(f"🤖 LLM выбрал: {selected.name}")
                    return selected
                else:
                    self.logger.info(f"✅ Найден подходящий агент: {capable_agents[0].name}")
                    return capable_agents[0]
            
            # Если никто не подходит - используем RAG
            self.logger.info(f"ℹ️ Нет подходящего агента, используем RAG")
            return self.default_agent
        
        # Гибридный режим - RAG + специализированные агенты
        if self.mode == "hybrid":
            # Сначала пытаемся найти специализированного агента
            for agent in self.agents:
                if agent != self.default_agent and agent.can_handle(query):
                    self.logger.info(f"✅ Найден специализированный агент: {agent.name}")
                    return agent
            
            # Иначе используем RAG
            self.logger.info(f"ℹ️ Используем RAG агента как fallback")
            return self.default_agent
        
        # Если режим не распознан - используем RAG
        return self.default_agent
    
    def _llm_select_agent(self, query: str, agents: List[BaseAgent]) -> BaseAgent:
        """Использует LLM для выбора агента"""
        agents_desc = "\n".join([
            f"- {a.name}: {a.description}" 
            for a in agents
        ])
        
        prompt = f"""
        Выбери наиболее подходящего агента для обработки запроса.
        
        Доступные агенты:
        {agents_desc}
        
        Запрос: {query}
        
        Ответ должен содержать только имя агента из списка выше.
        """
        
        try:
            response = llamaindex_llm.complete(prompt)
            chosen_name = response.text.strip()
            
            # Находим агента по имени
            for agent in agents:
                if agent.name.lower() in chosen_name.lower():
                    return agent
            
            # Если не нашли, возвращаем первого
            return agents[0]
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка выбора агента: {e}")
            return agents[0]
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        Обрабатывает запрос через подходящего агента
        """
        self.logger.info(f"📝 Обработка запроса: {query[:100]}...")
        
        # Сохраняем историю
        self.query_history.append({
            "query": query,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        })
        
        # Выбираем агента
        agent = self._select_agent(query)
        
        if not agent:
            return {
                "query": query,
                "result": "❌ Нет подходящего агента для обработки запроса",
                "agent": None,
                "status": "error"
            }
        
        self.logger.info(f"🎯 Выбран агент: {agent.name}")
        
        # Обрабатываем запрос
        try:
            result = agent.process(query)
        except Exception as e:
            self.logger.error(f"❌ Ошибка обработки: {e}")
            return {
                "query": query,
                "result": f"❌ Ошибка: {str(e)}",
                "agent": agent.name,
                "status": "error"
            }
        
        # Извлекаем ответ
        answer = result.get("result", {})
        if isinstance(answer, dict):
            answer_text = answer.get("answer", "Нет ответа")
        else:
            answer_text = str(answer)
        
        return {
            "query": query,
            "result": answer_text,
            "agent": agent.name,
            "status": result.get("status", "success"),
            "details": result
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику супервизора"""
        return {
            "mode": self.mode,
            "total_agents": len(self.agents),
            "agents": [a.get_info() for a in self.agents],
            "default_agent": self.default_agent.name if self.default_agent else None,
            "history_count": len(self.query_history),
            "status": "active"
        }
    
    def set_mode(self, mode: str):
        """Меняет режим работы супервизора"""
        if mode in ["rag_first", "router", "hybrid"]:
            self.mode = mode
            self.logger.info(f"🔄 Режим изменен на: {mode}")
        else:
            raise ValueError(f"Неизвестный режим: {mode}")

# Создаем глобальный экземпляр супервизора
supervisor = Supervisor(mode="rag_first")  # По умолчанию - RAG-first