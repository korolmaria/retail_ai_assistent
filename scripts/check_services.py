#!/usr/bin/env python3
"""
Простая проверка сервисов
"""
import requests
import sys

def check_service(name, url, expected_code=200):
    """Проверяет доступность сервиса"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == expected_code:
            return True, response
        return False, f"Код {response.status_code}"
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 50)
    print("🔍 ПРОВЕРКА СЕРВИСОВ")
    print("=" * 50)
    
    all_ok = True
    
    # 1. Neo4j
    ok, result = check_service("Neo4j", "http://localhost:7474")
    print(f"\n🗄️  Neo4j: {'✅ РАБОТАЕТ' if ok else '❌ НЕ РАБОТАЕТ'}")
    if not ok:
        print(f"   Ошибка: {result}")
        all_ok = False
    
    # 2. Qdrant
    ok, response = check_service("Qdrant", "http://localhost:6333/")
    print(f"\n🧩 Qdrant: {'✅ РАБОТАЕТ' if ok else '❌ НЕ РАБОТАЕТ'}")
    if ok:
        try:
            data = response.json()
            print(f"   Версия: {data.get('version', 'unknown')}")
            print(f"   Статус: {data.get('title', 'unknown')}")
            
            # Проверяем коллекции
            collections_resp = requests.get("http://localhost:6333/collections", timeout=5)
            if collections_resp.status_code == 200:
                collections = collections_resp.json().get('collections', [])
                if collections:
                    print(f"   Коллекции: {', '.join([c['name'] for c in collections])}")
                else:
                    print("   ⚠️ Коллекции не созданы")
        except:
            pass
    else:
        print(f"   Ошибка: {result}")
        all_ok = False
    
    # 3. LM Studio
    ok, response = check_service("LM Studio", "http://localhost:1234/v1/models")
    print(f"\n🤖 LM Studio: {'✅ РАБОТАЕТ' if ok else '❌ НЕ РАБОТАЕТ'}")
    if ok:
        try:
            data = response.json()
            models = data.get('data', [])
            if models:
                print(f"   Модели: {', '.join([m['id'] for m in models[:3]])}")
            else:
                print("   ⚠️ Модели не загружены")
        except:
            pass
    else:
        print(f"   Ошибка: {result}")
        all_ok = False
    
    print("\n" + "=" * 50)
    
    if all_ok:
        print("✅ ВСЕ СЕРВИСЫ РАБОТАЮТ!")
        print("\n📋 ДОСТУПНЫЕ ИНТЕРФЕЙСЫ:")
        print("   - Neo4j Browser:  http://localhost:7474")
        print("   - Qdrant:         http://localhost:6333")
        print("   - Qdrant Dashboard: http://localhost:6333/dashboard")
        print("   - LM Studio API:  http://localhost:1234")
        print("\n🚀 Запускайте приложение: python3 main.py")
    else:
        print("❌ НЕ ВСЕ СЕРВИСЫ ДОСТУПНЫ")
        print("💡 Исправьте ошибки и попробуйте снова")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())