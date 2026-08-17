import requests
from typing import Optional, Any, List
from src.config import LM_STUDIO_URL, LM_STUDIO_API_KEY, MAX_TOKENS, TEMPERATURE_BALANCED, TOP_P, MODEL_NAME

from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback

class LMStudioClient:
    def __init__(self, base_url: str = LM_STUDIO_URL):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LM_STUDIO_API_KEY}"
        }
    
    def generate(self, 
                 prompt: str,
                 system_prompt: Optional[str] = None,
                 max_tokens: int = MAX_TOKENS,
                 temperature: float = TEMPERATURE_BALANCED,
                 top_p: float = TOP_P,
                 **kwargs) -> str:
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=600
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Ошибка: {e}")
            return ""

# Создаем глобальный объект вашего клиента
lm_client = LMStudioClient()


# ============================================================================
# ОБЁРТКА ДЛЯ LLAMAINDEX (Чтобы ваш клиент строил графы в Neo4j)
# ============================================================================
class LlamaIndexLMStudioWrapper(CustomLLM):
    context_window: int = 4096
    num_output: int = MAX_TOKENS
    model_name: str = MODEL_NAME

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # Вызываем ваш нативный клиент!
        response_text = lm_client.generate(prompt=prompt, **kwargs)
        return CompletionResponse(text=response_text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("Стриминг в кастомной обертке не реализован")

# Этот объект мы отдадим в настройки LlamaIndex на следующем шаге
llamaindex_llm = LlamaIndexLMStudioWrapper()
