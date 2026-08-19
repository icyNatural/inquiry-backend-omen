from dataclasses import dataclass
from typing import Dict
import re


@dataclass
class SourcePlan:
    personal_memory: bool = False
    external_web: bool = False
    scientific_sources: bool = False
    model_knowledge: bool = False


@dataclass
class QueryAnalysis:
    original_query: str
    normalized_query: str
    changed: bool
    confidence: float
    question_type: str
    needs_personal_memory: bool
    needs_external_evidence: bool
    reasoning_depth: str
    source_plan: SourcePlan


# small deterministic normalization map for obvious typos
_NORMALIZATION_MAP = {
    "homeothersis": "homeorhesis",
    "homeorhesis": "homeorhesis",
}


def _simple_normalize(query: str) -> (str, bool, float):
    q = query.strip()
    lower = q.lower()

    # exact known misspelling
    tokenized = re.findall(r"\w+", lower)
    changed = False
    confidence = 0.9

    for i, tok in enumerate(tokenized):
        if tok in _NORMALIZATION_MAP:
            tokenized[i] = _NORMALIZATION_MAP[tok]
            changed = True
            confidence = 0.95

    if changed:
        # reconstruct with simple join to avoid altering punctuation
        normalized = " ".join(tokenized)
        return normalized, True, confidence

    # no deterministic change
    return q, False, confidence


def analyze_query(query: str) -> QueryAnalysis:
    orig = query or ""

    normalized, changed, confidence = _simple_normalize(orig)

    lower = normalized.lower()

    # deterministic classification heuristics (order matters)
    needs_personal = False
    needs_external = False
    qtype = "open_reasoning"
    reasoning = "light"

    # comparison
    if re.search(r"\bcompare\b", lower):
        qtype = "comparison"
        needs_personal = True if re.search(r"\b(my|my notes|my notes)\b", lower) else False
        needs_external = True
        reasoning = "deep"
    # personal recall patterns
    elif re.search(r"\b(what did i say|what have i|did i say)\b", lower):
        qtype = "personal_recall"
        needs_personal = True
        needs_external = False
        reasoning = "light"
    # planning
    elif re.search(r"\b(plan|how to|steps to|strategy)\b", lower):
        qtype = "planning"
        needs_external = True
        reasoning = "deep"
    # factual concept
    elif re.search(r"^\s*(what is|who is|when did|define|what are)\b", lower):
        qtype = "factual_concept"
        needs_external = True
        reasoning = "none"
    # synthesis keywords
    elif re.search(r"\b(synthesize|synthesis|connect|integrate)\b", lower):
        qtype = "synthesis"
        needs_personal = False
        needs_external = True
        reasoning = "deep"
    # fallback for open reasoning
    elif re.search(r"\b(why|how)\b", lower):
        qtype = "open_reasoning"
        reasoning = "deep"
        needs_external = False

    # Build source plan
    source_plan = SourcePlan(
        personal_memory=needs_personal,
        external_web=needs_external,
        scientific_sources=False,
        model_knowledge=not needs_external,
    )

    return QueryAnalysis(
        original_query=orig,
        normalized_query=normalized,
        changed=changed,
        confidence=confidence,
        question_type=qtype,
        needs_personal_memory=needs_personal,
        needs_external_evidence=needs_external,
        reasoning_depth=reasoning,
        source_plan=source_plan,
    )
