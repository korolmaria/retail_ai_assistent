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
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))
TOP_P = float(os.getenv("TOP_P", 0.85))
TEMPERATURE_CREATIVE = float(os.getenv("TEMPERATURE_CREATIVE", 0.8))
TEMPERATURE_ANALYTICAL = float(os.getenv("TEMPERATURE_ANALYTICAL", 0.1))
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
# НАСТРОЙКИ OCR
# ============================================================================
USE_OCR = True
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "rus+eng")
OCR_DPI = int(os.getenv("OCR_DPI", 600))
OCR_USE_EASYOCR = os.getenv("OCR_USE_EASYOCR", "True").lower() == "true"

# ============================================================================
# НАСТРОЙКИ ПАРСЕРА
# ============================================================================
PARSE_IMAGES = True
PARSE_TABLES = True
PARSE_HEADERS = True
KEEP_PAGE_STRUCTURE = True

# ============================================================================
# НАСТРОЙКИ ЧАНКОВ (CHUNKS)
# ============================================================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
SEARCH_K = int(os.getenv("SEARCH_K", 3))

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
    
    # OCR
    USE_OCR = USE_OCR
    OCR_LANGUAGE = OCR_LANGUAGE
    OCR_DPI = OCR_DPI
    OCR_USE_EASYOCR = OCR_USE_EASYOCR
    
    # Настройки парсера
    PARSE_IMAGES = PARSE_IMAGES
    PARSE_TABLES = PARSE_TABLES
    PARSE_HEADERS = PARSE_HEADERS
    KEEP_PAGE_STRUCTURE = KEEP_PAGE_STRUCTURE
    
    # Размеры чанков
    CHUNK_SIZE = CHUNK_SIZE
    CHUNK_OVERLAP = CHUNK_OVERLAP
    SEARCH_K = SEARCH_K
    
    # ================================================================
    # ПУТИ ДЛЯ ХРАНЕНИЯ - ВСЕ В CACHE
    # ================================================================
    BASE_DIR = BASE_DIR
    DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
    
    # Основная кэш-директория
    CACHE_DIR = BASE_DIR / "cache"
    
    # Поддиректории в cache
    PARSED_DOCS_DIR = CACHE_DIR / "parsed_documents"      # JSON с результатами парсинга
    EXTRACTED_IMAGES_DIR = CACHE_DIR / "extracted_images"  # Извлеченные изображения
    EXTRACTED_TABLES_DIR = CACHE_DIR / "extracted_tables"  # Извлеченные таблицы
    STORAGE_CONTEXT_DIR = CACHE_DIR / "storage_context"    # LlamaIndex StorageContext
    VECTOR_STORE_DIR = CACHE_DIR / "vector_store"          # Векторный индекс
    DOCSTORE_DIR = CACHE_DIR / "docstore"                  # DocStore
    INDEX_CACHE_DIR = CACHE_DIR / "index_cache"            # Кэш индексов
    CHUNKS_DIR = CACHE_DIR / "chunks"                      # ← НОВОЕ: чанки с метаданными
    
    # Файлы кэша
    PARSED_CACHE_FILE = CACHE_DIR / "parsed_cache.pkl"
    INDEX_CACHE_FILE = CACHE_DIR / "index_cache.pkl"
    METADATA_CACHE_FILE = CACHE_DIR / "metadata_cache.pkl"
    CHUNKS_CACHE_FILE = CACHE_DIR / "chunks_cache.pkl"     # ← НОВОЕ: кэш чанков
    
    # Модели
    MODELS_DIR = MODELS_DIR
    
    # Поддерживаемые расширения
    SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.md']

# ================================================================
# СОЗДАНИЕ ВСЕХ ДИРЕКТОРИЙ
# ================================================================
def ensure_directories():
    """Создает все необходимые директории"""
    dirs = [
        Config.DOCUMENTS_DIR,
        Config.CACHE_DIR,
        Config.PARSED_DOCS_DIR,
        Config.EXTRACTED_IMAGES_DIR,
        Config.EXTRACTED_TABLES_DIR,
        Config.STORAGE_CONTEXT_DIR,
        Config.VECTOR_STORE_DIR,
        Config.DOCSTORE_DIR,
        Config.INDEX_CACHE_DIR,
        Config.CHUNKS_DIR,  # ← НОВОЕ
        Config.MODELS_DIR,
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        # Создаем .gitkeep для пустых директорий
        gitkeep = d / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

ensure_directories()

# Экспортируем пути для удобства
DOCUMENTS_DIR = Config.DOCUMENTS_DIR
CACHE_DIR = Config.CACHE_DIR
PARSED_DOCS_DIR = Config.PARSED_DOCS_DIR
STORAGE_CONTEXT_DIR = Config.STORAGE_CONTEXT_DIR
DOCSTORE_DIR = Config.DOCSTORE_DIR
CHUNKS_DIR = Config.CHUNKS_DIR  # ← НОВОЕ

DEBUG = os.getenv("DEBUG", "True").lower() == "true"