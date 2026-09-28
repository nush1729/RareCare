"""WSGI entry point: `gunicorn wsgi:app` (Docker / HF Space) or `python wsgi.py` (local)."""

import logging
import os

from rarecare.app import create_app

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "7860")), debug=False)
