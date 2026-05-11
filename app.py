from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from ingest.query import query as rag_query


class AskRequest(BaseModel):
    question: str
    k: int = 4
    model: str = "claude-sonnet-4-6"


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Observable RAG", lifespan=lifespan)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    result = rag_query(
        req.question,
        k=req.k,
        model=req.model,
    )
    return AskResponse(answer=result.answer, sources=result.sources)
