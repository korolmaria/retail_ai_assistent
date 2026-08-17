# src/agents/api_agents.py
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
from src.agents.base_agent import BaseAgent

class OrdersAgent(BaseAgent):
    """Агент для работы с заказами и поставками"""
    
    def __init__(self):
        super().__init__(
            name="OrdersAgent",
            description="Агент для работы с заказами, поставками и статусами заказов",
            version="1.0.0"
        )
        # Заглушка данных
        self.mock_orders = [
            {
                "id": "ORD-001",
                "status": "доставлен",
                "date": "2026-01-15",
                "items": ["Товар А", "Товар Б"],
                "supplier": "ООО Поставщик"
            },
            {
                "id": "ORD-002",
                "status": "в обработке",
                "date": "2026-08-10",
                "items": ["Товар В"],
                "supplier": "ИП Иванов"
            },
            {
                "id": "ORD-003",
                "status": "ожидает",
                "date": "2026-08-12",
                "items": ["Товар Г", "Товар Д"],
                "supplier": "ООО ТехноСнаб"
            }
        ]
    
    def process(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Обработка запросов по заказам"""
        self.logger.info(f"📦 Запрос по заказам: {query}")
        
        # Заглушка - возвращаем мок-данные с имитацией анализа
        response = self._analyze_query(query)
        
        return {
            "agent": self.name,
            "query": query,
            "result": {
                "answer": response,
                "data": self.mock_orders,
                "source": "mock_api"
            },
            "status": "success"
        }
    
    def _analyze_query(self, query: str) -> str:
        """Анализирует запрос и формирует ответ (заглушка)"""
        query_lower = query.lower()
        
        if "статус" in query_lower or "где" in query_lower:
            # Поиск заказа по номеру
            import re
            order_id_match = re.search(r'ORD-\d{3}', query)
            if order_id_match:
                order_id = order_id_match.group()
                for order in self.mock_orders:
                    if order["id"] == order_id:
                        return f"Заказ {order_id} имеет статус: {order['status']}. Дата: {order['date']}"
            
            return "Вот список ваших заказов с текущими статусами. Для получения деталей уточните номер заказа."
        
        elif "поставк" in query_lower or "поставщик" in query_lower:
            suppliers = set(o["supplier"] for o in self.mock_orders)
            return f"Активные поставщики: {', '.join(suppliers)}. Всего заказов: {len(self.mock_orders)}"
        
        else:
            return f"В системе зарегистрировано {len(self.mock_orders)} заказов. Для получения деталей уточните запрос."
    
    def can_handle(self, query: str) -> bool:
        """Проверяет, может ли агент обработать запрос"""
        keywords = ["заказ", "поставк", "заказа", "поставщик", "отгрузк", "доставк", "статус заказ"]
        return any(kw in query.lower() for kw in keywords)


class GoodsAgent(BaseAgent):
    """Агент для работы с товарами и номенклатурой"""
    
    def __init__(self):
        super().__init__(
            name="GoodsAgent",
            description="Агент для работы с товарами, номенклатурой и остатками",
            version="1.0.0"
        )
        # Заглушка данных
        self.mock_goods = [
            {"id": "GOOD-001", "name": "Ноутбук", "category": "Электроника", "stock": 45},
            {"id": "GOOD-002", "name": "Мышь", "category": "Электроника", "stock": 120},
            {"id": "GOOD-003", "name": "Стул", "category": "Мебель", "stock": 15},
            {"id": "GOOD-004", "name": "Стол", "category": "Мебель", "stock": 8},
            {"id": "GOOD-005", "name": "Принтер", "category": "Офисная техника", "stock": 23}
        ]
    
    def process(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Обработка запросов по товарам"""
        self.logger.info(f"📦 Запрос по товарам: {query}")
        
        response = self._analyze_query(query)
        
        return {
            "agent": self.name,
            "query": query,
            "result": {
                "answer": response,
                "data": self.mock_goods,
                "source": "mock_api"
            },
            "status": "success"
        }
    
    def _analyze_query(self, query: str) -> str:
        """Анализирует запрос и формирует ответ (заглушка)"""
        query_lower = query.lower()
        
        if "остаток" in query_lower or "в наличии" in query_lower:
            total = sum(g["stock"] for g in self.mock_goods)
            return f"Общий остаток товаров на складе: {total} единиц. Всего позиций: {len(self.mock_goods)}"
        
        elif "категори" in query_lower:
            categories = set(g["category"] for g in self.mock_goods)
            return f"Категории товаров: {', '.join(categories)}"
        
        elif "поиск" in query_lower or "найти" in query_lower:
            # Пытаемся найти товар по названию
            for good in self.mock_goods:
                if good["name"].lower() in query_lower:
                    return f"Найден товар: {good['name']} (категория: {good['category']}), остаток: {good['stock']} шт."
            return "Товар не найден. Попробуйте уточнить название."
        
        else:
            return f"В системе {len(self.mock_goods)} товаров в {len(set(g['category'] for g in self.mock_goods))} категориях."
    
    def can_handle(self, query: str) -> bool:
        """Проверяет, может ли агент обработать запрос"""
        keywords = ["товар", "номенклатур", "остаток", "в наличии", "категори", "склад", "поиск товар"]
        return any(kw in query_lower for kw in keywords)


class ReportsAgent(BaseAgent):
    """Агент для работы с отчетами и аналитикой"""
    
    def __init__(self):
        super().__init__(
            name="ReportsAgent",
            description="Агент для генерации отчетов и аналитики",
            version="1.0.0"
        )
    
    def process(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Обработка запросов по отчетам"""
        self.logger.info(f"📊 Запрос по отчетам: {query}")
        
        # Заглушка
        return {
            "agent": self.name,
            "query": query,
            "result": {
                "answer": "Функция генерации отчетов находится в разработке. Доступны базовые отчеты по заказам и товарам.",
                "available_reports": ["Статусы заказов", "Остатки товаров", "Активные поставщики"],
                "source": "mock_api"
            },
            "status": "success"
        }
    
    def can_handle(self, query: str) -> bool:
        """Проверяет, может ли агент обработать запрос"""
        keywords = ["отчет", "аналитик", "статистик", "сводк", "обзор"]
        return any(kw in query_lower for kw in keywords)


class AnalyticsAgent(BaseAgent):
    """Агент для аналитики и прогнозов"""
    
    def __init__(self):
        super().__init__(
            name="AnalyticsAgent",
            description="Агент для аналитики, прогнозов и бизнес-показателей",
            version="1.0.0"
        )
    
    def process(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Обработка аналитических запросов"""
        self.logger.info(f"📈 Аналитический запрос: {query}")
        
        # Заглушка с имитацией аналитики
        response = self._generate_mock_analytics(query)
        
        return {
            "agent": self.name,
            "query": query,
            "result": {
                "answer": response,
                "metrics": {
                    "total_orders": 156,
                    "active_suppliers": 24,
                    "total_goods": 342,
                    "stock_value": "1.2M",
                    "period": "август 2026"
                },
                "source": "mock_api"
            },
            "status": "success"
        }
    
    def _generate_mock_analytics(self, query: str) -> str:
        """Генерирует мок-аналитику"""
        return """
        📊 Аналитика за август 2026:
        - Всего заказов: 156
        - Активных поставщиков: 24
        - Товаров на складе: 342 позиции
        - Общая стоимость запасов: 1.2M
        - Выполнено заказов: 87%
        
        Динамика: рост на 12% по сравнению с прошлым месяцем.
        """
    
    def can_handle(self, query: str) -> bool:
        """Проверяет, может ли агент обработать запрос"""
        keywords = ["аналитик", "прогноз", "показател", "метрик", "динамик", "тренд"]
        return any(kw in query_lower for kw in keywords)