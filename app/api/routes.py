from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.agent.orchestrator import Orchestrator
from app.schemas.models import QueryRequest, QueryResponse

log = logging.getLogger(__name__)

router = APIRouter()
agent = Orchestrator()


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        answer, context, steps = agent.run(req.question)
        return QueryResponse(answer=answer, context=context, steps=steps)
    except FileNotFoundError as e:
        # Typically means index not ingested yet.
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        log.exception("Query failed")
        raise HTTPException(status_code=500, detail="Query failed") from e

