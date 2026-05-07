from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings


def _configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


_configure_logging()

app = FastAPI(title="Enterprise AI Assistant", version="0.1.0")
app.include_router(router)
