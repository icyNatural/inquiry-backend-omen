from typing import Tuple

from app.models.inquiry import InquiryRequest
from app.models.investigation import InvestigationCreate
from app.repositories.investigation_repository import InvestigationRepository
from app.services.legacy_inquiry_engine import investigate


def run_and_persist(payload: InquiryRequest) -> Tuple[dict, object]:
    """
    Run the reasoning pipeline and persist an Investigation artifact.

    Returns a tuple: (reasoning_result_dict, persisted_investigation)
    """
    # Run the existing orchestrator (stateless reasoning)
    result = investigate(
        user_query=payload.query,
        mode=payload.mode,
        scope=payload.scope,
        knowledge_policy=payload.knowledge_policy,
    )

    # Build a persistable Investigation payload while preserving provenance
    # Basic observability/tracing
    stages = [
        "validated_query",
        "query_expansion",
        "retrieval",
        "scope_filter",
        "diversify",
        "rerank",
        "interpret",
        "llm_reasoning",
        "persistence",
    ]

    model_text = result.get("model")
    # Detect simple fallback pattern in model output
    answer_text = result.get("answer", "") or ""
    model_fallback = False
    fallback_to = None
    if isinstance(answer_text, str) and answer_text.startswith("[FALLBACK TO "):
        model_fallback = True
        # parse fallback model name
        try:
            fallback_to = answer_text.split("[FALLBACK TO ", 1)[1].split("]", 1)[0]
        except Exception:
            fallback_to = None

    brief = {
        "mode": payload.mode,
        "knowledge_policy": str(payload.knowledge_policy),
        "scope": payload.scope,
        "search_queries": result.get("search_queries"),
        "retrieved_count": result.get("retrieved_count"),
        "selected_count": result.get("selected_count"),
        "model": model_text,
        "model_fallback": model_fallback,
        "fallback_to": fallback_to,
        "stages": stages,
    }

    # Keep the full reasoning result in the report (can be large)
    report = {
        "answer": result.get("answer"),
        "organized_signals": result.get("organized_signals"),
        "sources": result.get("sources"),
        "search_queries": result.get("search_queries"),
    }

    inv_create = InvestigationCreate(
        original_question=payload.query,
        objective=None,
        brief=brief,
        report=report,
        confidence=None,
        parent_investigation_id=None,
    )

    repo = InvestigationRepository()
    investigation = repo.create(inv_create)

    return result, investigation
