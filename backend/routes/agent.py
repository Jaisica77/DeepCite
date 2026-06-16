from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from services.agent import run_agent
from services.retriever import get_retriever
from routes.upload import get_docs_store

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class AgentChatRequest(BaseModel):
    query: str
    doc_ids: List[str] = []
    chat_history: List[ChatMessage] = []
    use_web_search: bool = True


class Citation(BaseModel):
    type: str
    source: str
    title: Optional[str] = ""
    snippet: str


class AgentChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    tool_calls_made: int


@router.post("/agent-chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest):
    try:
        docs_store = get_docs_store()
        retriever = get_retriever()

        doc_metadata = [
            {"doc_id": k, "filename": v["filename"]}
            for k, v in docs_store.items()
        ]

        if request.doc_ids:
            from services.chunker import create_chunks
            retriever.clear()
            for doc_id in request.doc_ids:
                if doc_id not in docs_store:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Document {doc_id} not found"
                    )
                doc = docs_store[doc_id]
                text = doc.get("text", "")
                if not text:
                    continue
                chunks_dict = create_chunks(text, [512], overlap_percent=10)
                chunks = chunks_dict.get("512", [])
                retriever.add_documents(
                    chunks=chunks,
                    doc_id=doc_id,
                    chunk_size=512
                )

        history = [{"role": m.role, "content": m.content} for m in request.chat_history]

        result = run_agent(
            query=request.query,
            retriever=retriever,
            doc_metadata=doc_metadata,
            chat_history=history
        )

        return AgentChatResponse(
            answer=result["answer"],
            citations=[Citation(**c) for c in result["citations"]],
            tool_calls_made=result["tool_calls_made"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
