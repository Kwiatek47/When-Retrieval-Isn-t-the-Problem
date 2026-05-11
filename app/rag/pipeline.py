import logging
from time import perf_counter

from app.rag.models import PostRetrievalResult, RetrievalResult
from app.rag.post_retrieval import PostRetriever
from app.rag.pre_retrieval import PreRetriever
from app.rag.retrieval import MedicalKnowledgeRetriever
from app.schemas import ChatMessage


logger = logging.getLogger(__name__)


class RagPipeline:
    def __init__(
        self,
        *,
        pre_retriever: PreRetriever,
        retriever: MedicalKnowledgeRetriever,
        post_retriever: PostRetriever,
        retrieval_candidate_limit: int,
    ) -> None:
        self.pre_retriever = pre_retriever
        self.retriever = retriever
        self.post_retriever = post_retriever
        self.retrieval_candidate_limit = retrieval_candidate_limit

    async def run(self, *, messages: list[ChatMessage], system_prompt: str) -> PostRetrievalResult:
        started_at = perf_counter()
        pre_retrieval = await self.pre_retriever.prepare(messages)
        pre_retrieval_done_at = perf_counter()
        if pre_retrieval.requires_retrieval:
            retrieval = await self.retriever.retrieve(pre_retrieval, limit=self.retrieval_candidate_limit)
        else:
            retrieval = RetrievalResult(query=pre_retrieval, documents=[], provider="skipped")
        retrieval_done_at = perf_counter()

        result = self.post_retriever.assemble(
            messages=messages,
            system_prompt=system_prompt,
            pre_retrieval=pre_retrieval,
            retrieval=retrieval,
        )
        post_retrieval_done_at = perf_counter()

        logger.info(
            "rag_pipeline timing pre_retrieval=%.3fs retrieval=%.3fs post_retrieval=%.3fs total=%.3fs "
            "requires_retrieval=%s provider=%s candidate_documents=%d final_documents=%d",
            pre_retrieval_done_at - started_at,
            retrieval_done_at - pre_retrieval_done_at,
            post_retrieval_done_at - retrieval_done_at,
            post_retrieval_done_at - started_at,
            pre_retrieval.requires_retrieval,
            retrieval.provider,
            len(retrieval.documents),
            result.retrieval.documents_count,
        )
        return result
