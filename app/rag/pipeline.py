from app.rag.models import PostRetrievalResult, RetrievalResult
from app.rag.post_retrieval import PostRetriever
from app.rag.pre_retrieval import PreRetriever
from app.rag.retrieval import MedicalKnowledgeRetriever
from app.schemas import ChatMessage


class RagPipeline:
    def __init__(
        self,
        *,
        pre_retriever: PreRetriever,
        retriever: MedicalKnowledgeRetriever,
        post_retriever: PostRetriever,
        retrieval_limit: int,
    ) -> None:
        self.pre_retriever = pre_retriever
        self.retriever = retriever
        self.post_retriever = post_retriever
        self.retrieval_limit = retrieval_limit

    async def run(self, *, messages: list[ChatMessage], system_prompt: str) -> PostRetrievalResult:
        pre_retrieval = self.pre_retriever.prepare(messages)
        if pre_retrieval.requires_retrieval:
            retrieval = await self.retriever.retrieve(pre_retrieval, limit=self.retrieval_limit)
        else:
            retrieval = RetrievalResult(query=pre_retrieval, documents=[], provider="skipped")

        return self.post_retriever.assemble(
            messages=messages,
            system_prompt=system_prompt,
            pre_retrieval=pre_retrieval,
            retrieval=retrieval,
        )

