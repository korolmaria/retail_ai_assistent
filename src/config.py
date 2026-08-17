# src/config.py

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List

# Загружаем переменные из .env в корне проекта
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

# ============================================================================
# API НАСТРОЙКИ LM STUDIO
# ============================================================================
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "not-needed")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-7b-instruct")

# Параметры генерации
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2048))
TOP_P = float(os.getenv("TOP_P", 0.85))
TEMPERATURE_CREATIVE = float(os.getenv("TEMPERATURE_CREATIVE", 0.8))
TEMPERATURE_ANALYTICAL = float(os.getenv("TEMPERATURE_ANALYTICAL", 0.3))
TEMPERATURE_PRECISE = float(os.getenv("TEMPERATURE_PRECISE", 0.1))
TEMPERATURE_BALANCED = float(os.getenv("TEMPERATURE_BALANCED", 0.5))

# ============================================================================
# ИНФРАСТРУКТУРА БАЗ ДАННЫХ (DOCKER)
# ============================================================================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_secure_password_2026")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "retail_docs")

# ============================================================================
# НАСТРОЙКИ RAG МОДЕЛЕЙ
# ============================================================================
EMBEDDING_MODEL = str(MODELS_DIR / "embeddings" / "bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "mps")

# ============================================================================
# НАСТРОЙКИ OCR - ВКЛЮЧАЕМ!
# ============================================================================
USE_OCR = True  # ← ИСПРАВЛЕНО: было False
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "rus+eng")
OCR_DPI = int(os.getenv("OCR_DPI", 600))  # ← Увеличиваем для лучшего распознавания
OCR_USE_EASYOCR = os.getenv("OCR_USE_EASYOCR", "True").lower() == "true"

# ============================================================================
# КЛАСС CONFIG
# ============================================================================
class Config:
    """Конфигурация для RAG системы"""
    
    # API настройки
    LM_STUDIO_URL = LM_STUDIO_URL
    LM_STUDIO_API_KEY = LM_STUDIO_API_KEY
    MODEL_NAME = MODEL_NAME
    MAX_TOKENS = MAX_TOKENS
    TOP_P = TOP_P
    TEMPERATURE_CREATIVE = TEMPERATURE_CREATIVE
    TEMPERATURE_ANALYTICAL = TEMPERATURE_ANALYTICAL
    TEMPERATURE_PRECISE = TEMPERATURE_PRECISE
    TEMPERATURE_BALANCED = TEMPERATURE_BALANCED
    
    # Инфраструктура
    NEO4J_URI = NEO4J_URI
    NEO4J_USERNAME = NEO4J_USERNAME
    NEO4J_PASSWORD = NEO4J_PASSWORD
    QDRANT_URL = QDRANT_URL
    QDRANT_COLLECTION = QDRANT_COLLECTION
    
    # Модели
    EMBEDDING_MODEL = EMBEDDING_MODEL
    EMBEDDING_DEVICE = EMBEDDING_DEVICE
    
    # OCR - ВКЛЮЧАЕМ
    USE_OCR = USE_OCR  # ← ИСПРАВЛЕНО
    OCR_LANGUAGE = OCR_LANGUAGE
    OCR_DPI = OCR_DPI
    OCR_USE_EASYOCR = OCR_USE_EASYOCR
    
    # Размеры чанков
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
    SEARCH_K = int(os.getenv("SEARCH_K", 5))
    
    # ================================================================
    # ПУТИ ДЛЯ ХРАНЕНИЯ
    # ================================================================
    BASE_DIR = BASE_DIR
    DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
    QDRANT_STORAGE_DIR = BASE_DIR / "data" / "qdrant_storage"
    DOCSTORE_DIR = BASE_DIR / "data" / "docstore"
    CACHE_DIR = BASE_DIR / "cache"
    MODELS_DIR = MODELS_DIR
    
    # Поддерживаемые расширения
    SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.md']

# Создаем все необходимые директории
Config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
Config.QDRANT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
Config.DOCSTORE_DIR.mkdir(parents=True, exist_ok=True)
Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Экспортируем пути для удобства
DOCUMENTS_DIR = Config.DOCUMENTS_DIR
QDRANT_STORAGE_DIR = Config.QDRANT_STORAGE_DIR
DOCSTORE_DIR = Config.DOCSTORE_DIR
CACHE_DIR = Config.CACHE_DIR
MODELS_DIR = Config.MODELS_DIR
DEBUG = os.getenv("DEBUG", "True").lower() == "true"