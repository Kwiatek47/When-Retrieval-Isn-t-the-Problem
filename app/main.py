from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(title=settings.app_title, version=settings.app_version)

    application.include_router(api_router)
    application.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="static")

    return application


app = create_app()
