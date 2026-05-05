from contextlib import asynccontextmanager
from functools import lru_cache
from typing import AsyncIterator

from fastapi import Depends, FastAPI

from app.config import get_settings
from app.embedder import DocumentInput, MedCPTEmbedder
from app.schemas import EmbedDocumentsRequest, EmbedQueryRequest, EmbedResponse, HealthResponse


@lru_cache
def get_embedder() -> MedCPTEmbedder:
    settings = get_settings()
    return MedCPTEmbedder(
        model_name=settings.embedding_model_name,
        query_model_name=settings.query_model_name,
        document_model_name=settings.document_model_name,
        dimension=settings.embedding_dimension,
        query_max_length=settings.query_max_length,
        document_max_length=settings.document_max_length,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
    )


@asynccontextmanager
async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
    get_embedder()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    async def health(embedder: MedCPTEmbedder = Depends(get_embedder)) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model=embedder.model_name,
            query_model=embedder.query_model_name,
            document_model=embedder.document_model_name,
            dimension=embedder.dimension,
            device=str(embedder.device),
        )

    @application.post("/embed/documents", response_model=EmbedResponse)
    async def embed_documents(
        request: EmbedDocumentsRequest,
        embedder: MedCPTEmbedder = Depends(get_embedder),
    ) -> EmbedResponse:
        documents = [
            DocumentInput(title=document.title, text=document.text)
            for document in request.as_documents()
        ]
        embeddings = embedder.embed_documents(documents)
        return EmbedResponse(
            model=embedder.model_name,
            encoder="document",
            encoder_model=embedder.document_model_name,
            dimension=embedder.dimension,
            embeddings=embeddings,
        )

    @application.post("/embed/query", response_model=EmbedResponse)
    async def embed_query(
        request: EmbedQueryRequest,
        embedder: MedCPTEmbedder = Depends(get_embedder),
    ) -> EmbedResponse:
        embedding = embedder.embed_query(request.text)
        return EmbedResponse(
            model=embedder.model_name,
            encoder="query",
            encoder_model=embedder.query_model_name,
            dimension=embedder.dimension,
            embeddings=[embedding],
        )

    return application


app = create_app()
