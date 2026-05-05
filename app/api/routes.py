from typing import Annotated
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_llm_provider, get_telemetry_logger
from app.core.config import Settings, get_settings
from app.core.prompt_registry import resolve_prompt
from app.providers.base import LLMProvider, ProviderError, ProviderUnavailableError
from app.schemas import ChatRequest, ChatResponse, FeedbackRequest, FeedbackResponse
from app.services.chat_service import build_messages
from app.services.telemetry_service import TelemetryLogger


router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    llm_provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
    telemetry_logger: Annotated[TelemetryLogger, Depends(get_telemetry_logger)],
) -> ChatResponse:
    request_id = uuid4().hex
    timestamp = datetime.now(timezone.utc).isoformat()
    prompt_version, system_prompt = resolve_prompt(
        requested_version=request.prompt_version,
        active_version=settings.active_prompt_version,
        fallback_prompt=settings.system_prompt,
    )
    start = perf_counter()
    messages = build_messages(request.messages, system_prompt)

    try:
        provider_response = await llm_provider.chat(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
        )
        latency_ms = int((perf_counter() - start) * 1000)
        response = ChatResponse(
            model=provider_response.model,
            message=provider_response.message,
            done=provider_response.done,
            request_id=request_id,
            prompt_version=prompt_version,
            timestamp=timestamp,
            latency_ms=latency_ms,
        )
        telemetry_logger.log_chat_event(
            {
                "request_id": request_id,
                "status": "ok",
                "model": request.model,
                "temperature": request.temperature,
                "prompt_version": prompt_version,
                "messages": [message.model_dump() for message in request.messages],
                "response": response.model_dump(),
            }
        )
        return response
    except ProviderUnavailableError as exc:
        telemetry_logger.log_chat_event(
            {
                "request_id": request_id,
                "status": "provider_unavailable",
                "model": request.model,
                "temperature": request.temperature,
                "prompt_version": prompt_version,
                "messages": [message.model_dump() for message in request.messages],
                "error": str(exc),
            }
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ProviderError as exc:
        telemetry_logger.log_chat_event(
            {
                "request_id": request_id,
                "status": "provider_error",
                "model": request.model,
                "temperature": request.temperature,
                "prompt_version": prompt_version,
                "messages": [message.model_dump() for message in request.messages],
                "error": str(exc),
            }
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(
    request: FeedbackRequest,
    telemetry_logger: Annotated[TelemetryLogger, Depends(get_telemetry_logger)],
) -> FeedbackResponse:
    telemetry_logger.log_feedback_event(
        {
            "request_id": request.request_id,
            "rating": request.rating,
            "comment": request.comment,
            "model": request.model,
            "prompt_version": request.prompt_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    return FeedbackResponse(ok=True)
