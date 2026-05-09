import logging
import sys

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.core.config import get_settings


def configure_logging() -> None:
    app_logger = logging.getLogger("app")
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False
    if app_logger.handlers:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    app_logger.addHandler(handler)


configure_logging()


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(title=settings.app_title, version=settings.app_version)

    application.include_router(api_router)
    application.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="static")

    return application


app = create_app()
