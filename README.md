# Запускаем контейнеры
docker compose up -d

# Проверяем статус
docker compose ps
# Должны быть два контейнера: retail_neo4j_db и retail_qdrant_db

# Проверяем логи
docker compose logs

# Очистка Qdrant	
curl -X DELETE http://localhost:6333/collections/retail_docs

# Запуск	
python3 main.py

# Проверка Neo4j
curl -I http://localhost:7474

# Проверка Qdrant
curl -I http://localhost:6333/dashboard

# Проверить
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Что указано в Приложении 2 Информация о приемке товара?"}'