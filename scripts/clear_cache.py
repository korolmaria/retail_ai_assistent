# scripts/clear_cache.py

import shutil
from pathlib import Path
import sys
import logging

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clear_all_cache():
    """Полная очистка всех кэшей"""
    cache_dir = Config.CACHE_DIR
    
    if not cache_dir.exists():
        logger.info("📁 Кэш-директория не существует")
        return
    
    logger.info(f"🧹 Очистка кэша: {cache_dir}")
    
    # Список директорий для очистки
    dirs_to_clear = [
        Config.PARSED_DOCS_DIR,
        Config.EXTRACTED_IMAGES_DIR,
        Config.EXTRACTED_TABLES_DIR,
        Config.CHUNKS_DIR,
        Config.STORAGE_CONTEXT_DIR,
        Config.DOCSTORE_DIR,
        Config.INDEX_CACHE_DIR,
        Config.VECTOR_STORE_DIR,
    ]
    
    # Файлы для удаления
    files_to_remove = [
        Config.CACHE_DIR / "parsed_cache.pkl",
        Config.CACHE_DIR / "index_cache.pkl",
        Config.CACHE_DIR / "metadata_cache.pkl",
        Config.CACHE_DIR / "chunks_cache.pkl",
    ]
    
    # Очищаем директории
    for dir_path in dirs_to_clear:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            logger.info(f"  ✅ Удалена: {dir_path.name}")
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"  ✅ Создана пустая: {dir_path.name}")
    
    # Удаляем файлы
    for file_path in files_to_remove:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"  ✅ Удален файл: {file_path.name}")
    
    logger.info("✅ Кэш полностью очищен!")


def clear_qdrant_collection():
    """Очистка коллекции в Qdrant"""
    try:
        from qdrant_client import QdrantClient
        from src.config import Config
        
        client = QdrantClient(url=Config.QDRANT_URL)
        
        if client.collection_exists(Config.QDRANT_COLLECTION):
            client.delete_collection(Config.QDRANT_COLLECTION)
            logger.info(f"✅ Коллекция {Config.QDRANT_COLLECTION} удалена из Qdrant")
        else:
            logger.info(f"ℹ️ Коллекция {Config.QDRANT_COLLECTION} не существует")
            
    except Exception as e:
        logger.error(f"❌ Ошибка очистки Qdrant: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Очистка кэша RAG системы")
    parser.add_argument(
        "--qdrant",
        action="store_true",
        help="Очистить также коллекцию в Qdrant"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Принудительная очистка (без подтверждения)"
    )
    args = parser.parse_args()
    
    if not args.force:
        print("⚠️  ВНИМАНИЕ! Это удалит все кэшированные данные.")
        print(f"   Директория: {Config.CACHE_DIR}")
        if args.qdrant:
            print(f"   Коллекция Qdrant: {Config.QDRANT_COLLECTION}")
        response = input("Продолжить? (y/N): ")
        if response.lower() != 'y':
            print("❌ Отменено")
            sys.exit(0)
    
    clear_all_cache()
    
    if args.qdrant:
        clear_qdrant_collection()
    
    print("\n✅ Готово к переиндексации!")
    print(f"📁 Поместите документы в: {Config.DOCUMENTS_DIR}")
    print("🚀 Запустите: python scripts/index_documents.py")