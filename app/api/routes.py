from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_llm_provider
from app.core.config import Settings, get_settings
from app.providers.base import LLMProvider, ProviderError, ProviderUnavailableError
from app.schemas import ChatRequest, ChatResponse
from app.services.chat_service import build_messages


router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    llm_provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatResponse:
    try:
        return await llm_provider.chat(
            model=request.model,
            messages=build_messages(request.messages, settings.system_prompt),
            temperature=request.temperature,
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
