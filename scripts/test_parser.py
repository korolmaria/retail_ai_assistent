# scripts/test_parser.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_parser import document_parser
from src.config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_parser():
    """Тестирует парсер документов"""
    print("=" * 60)
    print("🧪 ТЕСТИРОВАНИЕ PARSER")
    print("=" * 60)
    
    # Проверяем директорию с документами
    docs_dir = Config.DOCUMENTS_DIR
    print(f"\n📁 Директория документов: {docs_dir}")
    print(f"   Существует: {docs_dir.exists()}")
    
    if not docs_dir.exists():
        print("❌ Директория не существует!")
        return False
    
    # Список файлов
    files = list(docs_dir.glob('*'))
    print(f"\n📄 Найдено файлов: {len(files)}")
    for f in files[:5]:  # Показываем первые 5
        print(f"   - {f.name} ({f.stat().st_size} байт)")
    
    if not files:
        print("❌ Нет файлов для парсинга!")
        return False
    
    # Парсим все документы
    print(f"\n🔄 Парсинг документов...")
    parsed = document_parser.parse_all_documents(docs_dir)
    
    print(f"\n📊 Результаты парсинга:")
    print(f"   Обработано документов: {len(parsed)}")
    
    if parsed:
        # Показываем первый документ
        first = parsed[0]
        print(f"\n📄 ПЕРВЫЙ ДОКУМЕНТ:")
        print(f"   Источник: {first['metadata']['source']}")
        print(f"   Тип: {first['metadata']['type']}")
        print(f"   Длина текста: {len(first['text'])} символов")
        print(f"   OCR текст: {'есть' if first['ocr_text'] else 'нет'}")
        print(f"   Таблиц: {len(first['tables'])}")
        print(f"   Картинки: {'есть' if first['has_images'] else 'нет'}")
        
        # Показываем первые 200 символов текста
        if first['text']:
            print(f"\n   Текст (первые 200 символов):")
            print(f"   {first['text'][:200]}...")
        
        # Проверяем, что текст не пустой
        if len(first['text']) > 100:
            print(f"\n✅ Текст успешно извлечен!")
        else:
            print(f"\n⚠️ Текст слишком короткий ({len(first['text'])} символов)")
    
    print("\n" + "=" * 60)
    return True

if __name__ == "__main__":
    success = test_parser()
    sys.exit(0 if success else 1)