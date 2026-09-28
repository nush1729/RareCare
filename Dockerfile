# Hugging Face Space (Docker SDK) / any container host. Serves on port 7860.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=7860
RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/rarecare

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app rarecare ./rarecare
COPY --chown=app:app configs ./configs
COPY --chown=app:app wsgi.py ./
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health')"
# One worker: sessions and rate limits are in-process (see rarecare/app/security.py).
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "60", "wsgi:app"]
