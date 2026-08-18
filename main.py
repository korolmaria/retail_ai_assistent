# main.py

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import uvicorn
from datetime import datetime

from src.init_rag import init_rag_system, get_rag_status
from src.rag_engine import rag_agent
from src.agents.supervisor import supervisor

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Агентская система с RAG",
    description="""
    Система с супервизором и агентами для обработки запросов.
    
    **Режимы работы супервизора:**
    - `rag_first`: всегда использовать RAG агента (текущий режим)
    - `router`: маршрутизация к специализированным агентам
    - `hybrid`: гибридный режим
    
    **Доступные агенты:**
    - RAGAgent: поиск в документах компании
    - OrdersAgent: работа с заказами и поставками
    - GoodsAgent: работа с товарами и номенклатурой
    - ReportsAgent: генерация отчетов
    - AnalyticsAgent: аналитика и прогнозы
    """,
    version="1.0.0"
)

# ============================================================================
# CORS (для фронтенда)
# ============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене заменить на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# МОДЕЛИ ДЛЯ ЗАПРОСОВ/ОТВЕТОВ
# ============================================================================

class ChatRequest(BaseModel):
    """Запрос к чату"""
    message: str
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    """Ответ от чата"""
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    agent: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    conversationId: Optional[str] = None
    processingTime: Optional[float] = None
    tokensUsed: Optional[int] = None
    structured: Optional[Dict[str, Any]] = None  # ← ДОБАВЛЕНО

class QueryRequest(BaseModel):
    """Запрос к супервизору"""
    query: str
    agent_type: Optional[str] = "auto"  # auto, rag, orders, goods, reports, analytics

class QueryResponse(BaseModel):
    """Ответ от супервизора"""
    query: str
    result: str
    agent: str
    status: str
    details: Optional[dict] = None

class ModeRequest(BaseModel):
    """Запрос на смену режима"""
    mode: str  # rag_first, router, hybrid

class RefreshRequest(BaseModel):
    """Запрос на обновление индекса"""
    force_reindex: bool = False

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ ПРИ СТАРТЕ
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Инициализация RAG при старте"""
    logger.info("🔄 Запуск инициализации RAG...")
    success = init_rag_system()
    if success:
        logger.info("✅ RAG система готова к работе")
    else:
        logger.warning("⚠️ RAG система не готова")
    
    # Выводим информацию о супервизоре
    stats = supervisor.get_stats()
    logger.info(f"📊 Супервизор: режим={stats['mode']}, агентов={stats['total_agents']}")

# ============================================================================
# КОРНЕВЫЕ ЭНДПОИНТЫ
# ============================================================================

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "service": "Агентская система с RAG",
        "status": "running",
        "version": "1.0.0",
        "supervisor_mode": supervisor.mode,
        "agents": [a.get_info() for a in supervisor.agents]
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# ОСНОВНОЙ ЧАТ - ЭНДПОИНТ /api/v1/chat
# ============================================================================

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    authorization: Optional[str] = Header(None),
    x_contractor_id: Optional[str] = Header(None),
    x_contractor_name: Optional[str] = Header(None)
):
    """
    Отправка сообщения в чат.
    Супервизор сам выберет подходящего агента.
    
    **Заголовки (опционально):**
    - `Authorization`: Bearer токен для авторизации
    - `X-Contractor-Id`: ID контрактора
    - `X-Contractor-Name`: Имя контрактора
    """
    try:
        logger.info(f"📝 Запрос: {request.message}")
        if x_contractor_id:
            logger.info(f"   Contractor ID: {x_contractor_id}")
        if x_contractor_name:
            logger.info(f"   Contractor Name: {x_contractor_name}")
        
        # Проверяем авторизацию (опционально)
        if authorization:
            # Здесь можно добавить проверку JWT токена
            # token = authorization.replace("Bearer ", "")
            # validate_token(token)
            pass
        
        # Проверяем инициализацию RAG
        if not rag_agent.is_initialized:
            logger.info("🔧 Инициализация RAG...")
            rag_agent.initialize_and_index()
        
        # Отправляем запрос через супервизор
        result = supervisor.process(request.message)
        
        # Извлекаем structured из результата
        structured = result.get('structured')
        
        # Формируем ответ
        return ChatResponse(
            success=result.get('status') != 'error',
            response=result.get('result', 'Нет ответа'),
            error=result.get('error'),
            agent=result.get('agent'),
            sources=result.get('details', {}).get('result', {}).get('sources'),
            conversationId=result.get('details', {}).get('conversationId'),
            processingTime=result.get('details', {}).get('processingTime'),
            tokensUsed=result.get('details', {}).get('tokensUsed'),
            structured=structured
        )
        
    except Exception as e:
        logger.error(f"❌ Ошибка обработки запроса: {e}", exc_info=True)
        return ChatResponse(
            success=False,
            error=str(e)
        )

# ============================================================================
# ПОТОКОВЫЙ ЧАТ - ЭНДПОИНТ /api/v1/chat/stream
# ============================================================================

@app.post("/api/v1/chat/stream")
async def chat_stream(
    request: ChatRequest,
    authorization: Optional[str] = Header(None),
    x_contractor_id: Optional[str] = Header(None),
    x_contractor_name: Optional[str] = Header(None)
):
    """
    Отправка сообщения с потоковым ответом (Server-Sent Events).
    """
    from fastapi.responses import StreamingResponse
    import json
    import asyncio
    
    async def generate():
        try:
            logger.info(f"📝 Стрим запрос: {request.message}")
            
            # Проверяем инициализацию RAG
            if not rag_agent.is_initialized:
                rag_agent.initialize_and_index()
            
            # Отправляем запрос через супервизор
            result = supervisor.process(request.message)
            
            # Формируем ответ
            response_text = result.get('result', 'Нет ответа')
            agent_name = result.get('agent', 'unknown')
            
            # Извлекаем structured
            structured = None
            if result.get('details') and isinstance(result['details'], dict):
                structured = result['details'].get('structured')
            
            # Отправляем по частям (имитация стрима)
            chunks = response_text.split('. ')
            for i, chunk in enumerate(chunks):
                if chunk:
                    # Добавляем точку обратно (кроме последней)
                    text = chunk + ('. ' if i < len(chunks) - 1 else '')
                    
                    # Отправляем чанк
                    yield f"data: {json.dumps({'content': text, 'chunk_index': i})}\n\n"
                    
                    # Небольшая задержка для эффекта стрима
                    await asyncio.sleep(0.1)
            
            # Отправляем метаданные
            yield f"data: {json.dumps({'metadata': {'agent': agent_name, 'conversationId': result.get('details', {}).get('conversationId'), 'structured': structured}})}\n\n"
            
            # Сигнал завершения
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"❌ Ошибка стрима: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# ============================================================================
# СТАРЫЕ ЭНДПОИНТЫ (СОХРАНЕНЫ ДЛЯ СОВМЕСТИМОСТИ)
# ============================================================================

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Задать вопрос через супервизор (устаревший эндпоинт).
    Используйте /api/v1/chat вместо этого.
    
    - **query**: Текст вопроса
    - **agent_type**: Тип агента (auto, rag, orders, goods, reports, analytics)
    """
    try:
        logger.info(f"📝 Запрос (устаревший /ask): {request.query}")
        
        # Если указан конкретный агент, используем его
        if request.agent_type != "auto":
            for agent in supervisor.agents:
                if agent.name.lower() == request.agent_type.lower():
                    result = agent.process(request.query)
                    return QueryResponse(
                        query=request.query,
                        result=result.get("result", {}).get("answer", "Нет результата") if isinstance(result.get("result"), dict) else str(result.get("result", "Нет результата")),
                        agent=agent.name,
                        status=result.get("status", "error"),
                        details=result
                    )
            
            # Если агент не найден, но запрошен конкретный
            available = [a.name for a in supervisor.agents]
            raise HTTPException(
                status_code=404, 
                detail=f"Агент {request.agent_type} не найден. Доступны: {', '.join(available)}"
            )
        
        # Автоматический выбор агента через супервизор
        result = supervisor.process(request.query)
        
        return QueryResponse(
            query=request.query,
            result=result.get("result", "Нет результата"),
            agent=result.get("agent", "unknown"),
            status=result.get("status", "error"),
            details=result.get("details")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка обработки запроса: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# RAG ЭНДПОИНТЫ
# ============================================================================

@app.get("/rag/stats")
async def get_rag_stats():
    """Получить статистику RAG системы"""
    try:
        return {
            "status": "success",
            "stats": get_rag_status()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/refresh")
async def refresh_rag(request: RefreshRequest = None):
    """Принудительное обновление индекса RAG"""
    try:
        force = request.force_reindex if request else False
        logger.info(f"🔄 Обновление индекса (force_reindex={force})")
        
        if force:
            rag_agent.initialize_and_index(force_reindex=True)
        else:
            rag_agent.initialize_and_index(force_reindex=False)
        
        return {
            "status": "success",
            "message": f"Индекс обновлен (force_reindex={force})",
            "stats": get_rag_status()
        }
    except Exception as e:
        logger.error(f"❌ Ошибка обновления индекса: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/query/test")
async def test_rag_query():
    """Тестовый запрос к RAG системе"""
    try:
        test_queries = [
            "Что такое политика работы с поставщиками?",
            "Какие правила оформления заказов?",
            "Как осуществляется отбор поставщиков?"
        ]
        
        results = []
        for query in test_queries:
            result = rag_agent.query(query)
            results.append({
                "query": query,
                "answer": result.get("answer", "Нет ответа")[:200] + "..." if len(result.get("answer", "")) > 200 else result.get("answer", ""),
                "sources_count": len(result.get("sources", [])),
                "structured": result.get("structured")  # ← ДОБАВЛЕНО
            })
        
        return {
            "status": "success",
            "results": results,
            "stats": get_rag_status()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# АГЕНТЫ ЭНДПОИНТЫ
# ============================================================================

@app.get("/agents")
async def list_agents():
    """Список всех агентов с их возможностями"""
    return {
        "status": "success",
        "mode": supervisor.mode,
        "agents": [
            {
                "name": a.name,
                "description": a.description,
                "type": a.__class__.__name__,
                "is_active": a.is_active,
                "version": getattr(a, 'version', '1.0.0')
            }
            for a in supervisor.agents
        ]
    }

@app.post("/supervisor/mode")
async def set_supervisor_mode(request: ModeRequest):
    """
    Изменить режим работы супервизора
    
    - `rag_first`: всегда использовать RAG агента
    - `router`: маршрутизировать к специализированным агентам
    - `hybrid`: гибридный режим
    """
    try:
        supervisor.set_mode(request.mode)
        return {
            "status": "success",
            "message": f"Режим изменен на: {request.mode}",
            "current_mode": supervisor.mode
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/supervisor/stats")
async def get_supervisor_stats():
    """Получить статистику супервизора"""
    try:
        stats = supervisor.get_stats()
        # Добавляем примеры запросов
        stats["recent_queries"] = supervisor.query_history[-5:] if supervisor.query_history else []
        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# АВТОРИЗАЦИЯ (ЗАГЛУШКА)
# ============================================================================

@app.post("/api/v1/auth/login")
async def login(request: Dict[str, str]):
    """Авторизация (заглушка)"""
    return {
        "success": True,
        "access_token": "mock_token_12345",
        "refresh_token": "mock_refresh_token_67890",
        "user": {
            "id": "1",
            "email": request.get("email", "user@example.com"),
            "name": "Тестовый Пользователь",
            "contractorId": "test_contractor_123",
            "contractorName": "ООО Тестовая Компания",
            "roles": ["user"]
        }
    }

@app.get("/api/v1/auth/me")
async def get_current_user():
    """Получение текущего пользователя (заглушка)"""
    return {
        "id": "1",
        "email": "user@example.com",
        "name": "Тестовый Пользователь",
        "contractorId": "test_contractor_123",
        "contractorName": "ООО Тестовая Компания",
        "roles": ["user"]
    }

# ============================================================================
# ГИБРИДНЫЙ ПОИСК - ДОПОЛНИТЕЛЬНЫЕ ЭНДПОИНТЫ
# ============================================================================

@app.get("/rag/hybrid/stats")
async def get_hybrid_stats():
    """Получить статистику гибридного поиска"""
    try:
        stats = rag_agent.get_stats()
        return {
            "status": "success",
            "hybrid_search": True,
            "stats": stats,
            "methods": stats.get('methods', ['vector', 'keyword', 'bm25']),
            "use_graph": stats.get('use_graph', False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/hybrid/weights")
async def set_hybrid_weights(
    vector: Optional[float] = None,
    keyword: Optional[float] = None,
    bm25: Optional[float] = None,
    graph: Optional[float] = None
):
    """
    Установить веса для методов гибридного поиска
    
    - vector: вес векторного поиска (по умолчанию 1.0)
    - keyword: вес TF-IDF поиска (по умолчанию 0.7)
    - bm25: вес BM25 поиска (по умолчанию 0.8)
    - graph: вес графового поиска (по умолчанию 0.5)
    """
    try:
        rag_agent.set_search_weights(vector, keyword, bm25, graph)
        return {
            "status": "success",
            "message": "Веса обновлены",
            "weights": rag_agent.hybrid_retriever.weights if rag_agent.hybrid_retriever else {}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/hybrid/test")
async def test_hybrid_search():
    """Тест гибридного поиска с разными методами"""
    try:
        test_queries = [
            "Что такое политика работы с поставщиками?",
            "Какие правила оформления заказов?",
            "Как осуществляется отбор поставщиков?"
        ]
        
        results = []
        for query in test_queries:
            result = rag_agent.query(query)
            results.append({
                "query": query,
                "answer": result.get("answer", "Нет ответа")[:200] + "..." if len(result.get("answer", "")) > 200 else result.get("answer", ""),
                "sources_count": result.get("sources_count", 0),
                "methods_used": result.get("methods_used", ['vector']),
                "hybrid_search": result.get("hybrid_search", False),
                "elapsed": result.get("elapsed", 0),
                "structured": result.get("structured")  # ← ДОБАВЛЕНО
            })
        
        return {
            "status": "success",
            "results": results,
            "stats": rag_agent.get_stats()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)