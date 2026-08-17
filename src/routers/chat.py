# src/routers/chat.py
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(
    prefix="/chat",
    tags=["Chat AI Engine"]
)

class ChatPayload(BaseModel):
    message: str

@router.post("")
async def handle_chat_message(request: Request, payload: ChatPayload, authorization: str = Header(None)):
    """
    Эндпоинт принимает сообщение от React, извлекает JWT из заголовков 
    и запускает мультиагентный граф LangGraph через состояние приложения.
    """
    # Проверяем сквозной JWT-токен от React-приложения
    if not authorization:
        raise HTTPException(
            status_code=401, 
            detail="Сессия не авторизована. JWT токен отсутствует в Headers (Authorization)."
        )

    # Достаем скомпилированный LangGraph из глобального состояния FastAPI приложения
    compiled_graph = request.app.state.compiled_graph
    
    if not compiled_graph:
        raise HTTPException(
            status_code=500,
            detail="ИИ-граф не инициализирован на сервере."
        )

    # Инициализируем стартовое состояние сессии для LangGraph
    initial_state = {
        "user_message": payload.message,
        "jwt_token": authorization,
        "next_node": "",
        "final_answer": None
    }

    try:
        # Прогоняем состояние через наш мультиагентный граф
        result_state = compiled_graph.invoke(initial_state)
        return {"response": result_state["final_answer"]}
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Внутренняя ошибка ИИ-роутера: {str(e)}"
        )
