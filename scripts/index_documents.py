# scripts/index_documents.py

import sys
from pathlib import Path
import logging
import time
import shutil
import subprocess
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.rag_engine import rag_agent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_docker_qdrant():
    """Проверка Qdrant через Docker"""
    try:
        result = subprocess.run(
            ['docker', 'ps', '--filter', 'name=qdrant', '--format', '{{.Status}}'],
            capture_output=True,
            text=True
        )
        
        if 'Up' in result.stdout:
            logger.info("✅ Qdrant контейнер запущен")
            return True
        else:
            logger.warning("⚠️ Qdrant контейнер не запущен")
            return False
    except Exception as e:
        logger.warning(f"⚠️ Ошибка проверки Docker: {e}")
        return False


def start_docker_qdrant():
    """Запуск Qdrant через Docker Compose"""
    logger.info("🚀 Запуск Qdrant через Docker Compose...")
    
    try:
        compose_file = Path("docker-compose.yml")
        if not compose_file.exists():
            logger.error("❌ docker-compose.yml не найден!")
            return False
        
        result = subprocess.run(
            ['docker-compose', 'up', '-d', 'qdrant'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            logger.info("✅ Qdrant запущен через Docker Compose")
            time.sleep(5)
            return True
        else:
            logger.error(f"❌ Ошибка запуска: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        return False


def check_qdrant():
    """Проверка Qdrant"""
    from qdrant_client import QdrantClient
    
    try:
        client = QdrantClient(url=Config.QDRANT_URL)
        
        try:
            collections = client.get_collections()
            logger.info(f"✅ Qdrant доступен, коллекций: {len(collections.collections)}")
            return True
        except:
            try:
                client.collection_exists(Config.QDRANT_COLLECTION)
                logger.info("✅ Qdrant доступен")
                return True
            except:
                return False
                
    except Exception as e:
        logger.warning(f"⚠️ Qdrant не доступен: {e}")
        return False


def ensure_qdrant():
    """Обеспечивает работу Qdrant"""
    if check_qdrant():
        return True
    
    if start_docker_qdrant():
        if check_qdrant():
            return True
    
    logger.error("❌ Не удалось запустить Qdrant")
    logger.info("\n💡 Попробуйте вручную:")
    logger.info("  docker-compose up -d qdrant")
    return False


def clear_cache():
    """Очистка кэша"""
    logger.info("🧹 Очистка кэша...")
    
    dirs_to_clear = ['chunks', 'storage_context', 'docstore', 'index_cache', 'parsed_documents']
    for dir_name in dirs_to_clear:
        dir_path = Config.CACHE_DIR / dir_name
        if dir_path.exists():
            shutil.rmtree(dir_path)
            logger.info(f"  🗑️ Удалена: {dir_name}")
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"  ✅ Создана: {dir_name}")
    
    files_to_remove = ['index_cache.pkl', 'metadata_cache.pkl', 'parsed_cache.pkl', 'chunks_cache.pkl']
    for file_name in files_to_remove:
        file_path = Config.CACHE_DIR / file_name
        if file_path.exists():
            file_path.unlink()
            logger.info(f"  🗑️ Удален: {file_name}")
    
    logger.info("✅ Кэш очищен!")


def clear_qdrant_collection():
    """Удаление коллекции в Qdrant"""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=Config.QDRANT_URL)
        
        if client.collection_exists(Config.QDRANT_COLLECTION):
            client.delete_collection(Config.QDRANT_COLLECTION)
            logger.info(f"🗑️ Коллекция {Config.QDRANT_COLLECTION} удалена")
        else:
            logger.info(f"ℹ️ Коллекция {Config.QDRANT_COLLECTION} не существует")
    except Exception as e:
        logger.warning(f"⚠️ Ошибка удаления коллекции: {e}")


def index_documents(force_reindex: bool = True, clear_first: bool = False):
    """Индексация документов"""
    logger.info("=" * 60)
    logger.info("🚀 ЗАПУСК ИНДЕКСАЦИИ ДОКУМЕНТОВ")
    logger.info("=" * 60)
    
    logger.info(f"📁 Директория документов: {Config.DOCUMENTS_DIR}")
    
    if clear_first:
        clear_cache()
        clear_qdrant_collection()
    
    # Проверяем Qdrant
    if not ensure_qdrant():
        logger.error("❌ Индексация отменена - Qdrant не доступен")
        logger.info("\n💡 Попробуйте:")
        logger.info("  docker-compose up -d qdrant")
        logger.info("  или")
        logger.info("  python3 scripts/index_local.py (без Qdrant)")
        return
    
    # Проверяем документы
    documents = list(Config.DOCUMENTS_DIR.glob("*.*"))
    documents = [d for d in documents if d.name != '.gitkeep']
    
    if not documents:
        logger.error("❌ Нет документов в директории!")
        logger.info(f"📁 Поместите документы в: {Config.DOCUMENTS_DIR}")
        return
    
    logger.info(f"📄 Найдено документов: {len(documents)}")
    for doc in documents:
        size_kb = doc.stat().st_size / 1024
        logger.info(f"   - {doc.name} ({size_kb:.1f} KB)")
    
    # Запускаем индексацию
    try:
        logger.info("\n🔄 Запуск индексации...")
        rag_agent.initialize_and_index(force_reindex=force_reindex)
        
        time.sleep(2)
        
        stats = rag_agent.get_stats()
        logger.info("\n" + "=" * 60)
        logger.info("📊 СТАТИСТИКА ПОСЛЕ ИНДЕКСАЦИИ")
        logger.info("=" * 60)
        logger.info(f"   ✅ Инициализирован: {stats.get('is_initialized', False)}")
        logger.info(f"   📚 Чанков: {stats.get('chunks_count', 0)}")
        logger.info(f"   📄 Документов в docstore: {stats.get('documents_in_docstore', 0)}")
        logger.info(f"   🗄️ Точки в Qdrant: {stats.get('qdrant_points', 0)}")
        logger.info(f"   🔍 Search K: {stats.get('search_k', 5)}")
        logger.info(f"   💾 Кэш: {stats.get('cache_dir', 'N/A')}")
        
        logger.info("\n✅ Индексация завершена успешно!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка индексации: {e}", exc_info=True)


def test_queries():
    """Тестовые запросы"""
    logger.info("\n" + "=" * 60)
    logger.info("🧪 ТЕСТОВЫЕ ЗАПРОСЫ")
    logger.info("=" * 60)
    
    test_questions = [
        "Какие требования к паллетам?",
        "Какие сроки годности для продуктов питания?",
        "Что нужно для заключения договора?",
    ]
    
    for question in test_questions:
        logger.info(f"\n❓ {question}")
        try:
            result = rag_agent.query(question)
            
            if result.get('status') == 'success':
                answer = result.get('answer', '')
                if len(answer) > 500:
                    answer = answer[:500] + "..."
                print(f"\n📝 {answer}")
                print(f"\n📚 Источников: {result.get('sources_count', 0)}")
                if result.get('sources'):
                    for i, src in enumerate(result['sources'][:3], 1):
                        src_name = src.get('metadata', {}).get('source', 'unknown')
                        src_page = src.get('page', '?')
                        print(f"   {i}. {src_name}, стр. {src_page}")
            else:
                print(f"❌ {result.get('answer', 'Ошибка')}")
        except Exception as e:
            logger.error(f"❌ Ошибка запроса: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Индексация документов")
    parser.add_argument(
        "--no-force",
        action="store_true",
        help="Не переиндексировать, использовать существующий кэш"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Выполнить тестовые запросы после индексации"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Очистить кэш перед индексацией"
    )
    args = parser.parse_args()
    
    # Индексация
    index_documents(
        force_reindex=not args.no_force,
        clear_first=args.clear
    )
    
    # Тесты
    if args.test:
        test_queries()