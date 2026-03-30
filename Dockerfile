FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    SNUG_BOT_DATA_DIR=/app/data

WORKDIR /app

# pdf2image requires pdftoppm from poppler-utils.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        poppler-utils \
        tini \
        ca-certificates \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY telegram_bot ./telegram_bot
COPY arxiv ./arxiv
COPY start.sh ./start.sh

RUN mkdir -p /app/data \
    && chmod +x /app/start.sh

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "telegram_bot.bot"]
