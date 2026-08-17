# src/agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Базовый класс для всех агентов"""
    
    def __init__(self, name: str, description: str, version: str = "1.0.0"):
        self.name = name
        self.description = description
        self.version = version
        self.logger = logging.getLogger(f"agent.{name}")
        self.is_active = True
    
    @abstractmethod
    def process(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Обработка запроса"""
        pass
    
    def can_handle(self, query: str) -> bool:
        """Проверяет, может ли агент обработать запрос"""
        return False  # По умолчанию агенты не могут обработать
    
    def get_info(self) -> Dict[str, Any]:
        """Возвращает информацию об агенте"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "is_active": self.is_active,
            "type": self.__class__.__name__
        }