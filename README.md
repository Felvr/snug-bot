# SNUG Telegram Bot (Bothost deploy bundle)

Минимальный набор файлов для запуска SNUG Telegram бота на bothost.ru через Git deploy.

## Что внутри

- `telegram_bot/` - код Telegram-бота
- `arxiv/unified_furniture_pipeline.py` - основной PDF/VLM pipeline
- `arxiv/furniture_postprocess.py` - постобработка таблиц
- `requirements.txt` - Python зависимости
- `.env.example` - шаблон переменных окружения (без секретов)
- `start.sh` - команда запуска
- `Dockerfile` - контейнер с установленным poppler (`pdftoppm`)
- `.dockerignore` - исключения для быстрой и безопасной сборки

## Важно по зависимостям

Для `pdf2image` нужен системный `poppler` (`pdftoppm`).
Если сборка на хосте упирается в отсутствие `poppler`, есть 2 варианта:

1. Перейти на тариф/режим с Docker и установить `poppler` в образе.
2. Уточнить у поддержки bothost возможность установки системного пакета в runtime.

## Локальная проверка

```bash
cp .env.example .env
# Заполни TELEGRAM_BOT_TOKEN и один из API key
pip install -r requirements.txt
bash start.sh
```

## Локальная проверка через Docker

```bash
docker build -t snug-bot:latest .
docker run --rm \
	-e TELEGRAM_BOT_TOKEN=... \
	-e OPENROUTER_API_KEY=... \
	-e SNUG_BOT_DATA_DIR=/app/data \
	snug-bot:latest
```

Если команда сборки без точки в конце, Docker вернет ошибку. Нужен именно контекст сборки `.`.

## Деплой на GitHub

```bash
git init
git add .
git commit -m "Initial bothost deploy bundle"
git branch -M main
git remote add origin https://github.com/<YOUR_USER>/<YOUR_REPO>.git
git push -u origin main
```

## Подключение к bothost

1. Создай новый бот-проект в bothost.
2. Подключи GitHub-репозиторий и выбери ветку `main`.
3. Если bothost запускает обычный runtime:
	- Build command: `pip install -r requirements.txt`
	- Start command: `bash start.sh`
4. Если bothost запускает через Docker:
	- Используй `Dockerfile` из репозитория (build/start команды обычно не нужны).
5. В разделе Env Variables добавь переменные из `.env.example`.

## Как передавать API безопасно (без git)

Никогда не коммить `.env` с реальными ключами.

Используй только env-переменные в bothost:

- `TELEGRAM_BOT_TOKEN`
- `VLM_API_KEY` или `OPENROUTER_API_KEY` или `OPENAI_API_KEY`
- остальные `SNUG_*` по необходимости

Код уже это поддерживает: `telegram_bot/common.py` читает `.env`, но если переменная уже задана в окружении, не перезаписывает ее.
То есть переменные из панели bothost будут использованы автоматически.

## Рекомендуемый минимум env для bothost

```dotenv
TELEGRAM_BOT_TOKEN=...
OPENROUTER_API_KEY=...
SNUG_BOT_DATA_DIR=/app/data
SNUG_BOT_POLL_TIMEOUT=30
SNUG_BOT_RETRY_DELAY_SEC=5
SNUG_BOT_PROGRESS_INTERVAL_SEC=45
SNUG_PIPELINE_TWO_PHASE=1
SNUG_PIPELINE_PAGE_STRATEGY=all
SNUG_PIPELINE_ROUTING_STRATEGY=all
SNUG_PIPELINE_SAMPLE_PAGES=8
SNUG_PIPELINE_ROUTING_SAMPLE_PAGES=32
SNUG_PIPELINE_DPI=170
SNUG_PIPELINE_ROUTING_DPI=120
VLM_BASE_URL=https://openrouter.ai/api/v1
VLM_MODEL=qwen/qwen3-vl-235b-a22b-instruct
VLM_ROUTER_MODEL=x-ai/grok-4.1-fast
```

## Безопасность

- Если ключи/токены уже светились в репозитории - обязательно сделай rotate (перевыпуск).
- Включи `.env` в `.gitignore`.
- Права доступа к GitHub-репо: лучше private.
