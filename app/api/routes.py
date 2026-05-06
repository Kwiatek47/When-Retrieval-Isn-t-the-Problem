from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_llm_provider, get_rag_pipeline
from app.core.config import Settings, get_settings
from app.providers.base import LLMProvider, ProviderError, ProviderUnavailableError
from app.rag.pipeline import RagPipeline
from app.schemas import ChatMessage, ChatRequest, ChatResponse


router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    llm_provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    rag_pipeline: Annotated[RagPipeline, Depends(get_rag_pipeline)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatResponse:
    try:
        rag_result = await rag_pipeline.run(messages=request.messages, system_prompt=settings.system_prompt)
        if rag_result.retrieval and rag_result.retrieval.status == "no_sources":
            return ChatResponse(
                model=request.model,
                message=ChatMessage(
                    role="assistant",
                    content=(
                        "Baza wiedzy nie zwróciła źródeł dla tego pytania, "
                        "więc nie mogę udzielić odpowiedzi opartej na cytowanych danych. "
                        "Skonsultuj decyzje medyczne z wykwalifikowanym lekarzem."
                    ),
                ),
                done=True,
                citations=rag_result.citations,
                retrieval=rag_result.retrieval,
            )

        llm_response = await llm_provider.chat(
            model=request.model,
            messages=rag_result.messages,
            temperature=request.temperature,
        )
        return ChatResponse(
            model=llm_response.model,
            message=llm_response.message,
            done=llm_response.done,
            citations=rag_result.citations,
            retrieval=rag_result.retrieval,
        )
    except ProviderUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
