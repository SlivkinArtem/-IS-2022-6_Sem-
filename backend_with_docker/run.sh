#!/bin/bash

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Ошибка: OPENAI_API_KEY не установлен"
    echo "Пожалуйста, создайте файл .env укажите ваш API ключ"
    exit 1
fi

echo "🚀 Запускаем сервисы..."
docker-compose up -d

echo "✅ Запуск завершен! Приложение доступно по адресу: http://localhost:8000"
echo "📝 Документация API: http://localhost:8000/docs"
echo "📊 Альтернативная документация: http://localhost:8000/redoc"