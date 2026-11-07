FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
RUN groupadd --system app && useradd --system --gid app --home /app app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[ml]"
COPY configs ./configs
COPY artifacts ./artifacts
RUN chown -R app:app /app

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["finqa-rag", "serve-agent", "--host", "0.0.0.0", "--port", "8000"]
