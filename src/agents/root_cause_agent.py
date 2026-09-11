"""Root-Cause Agent -- RAG over a curated casting-defect knowledge base.

IMPORTANT: like the Characterization Agent this chains after, there is no
defect-subtype ground truth in this dataset (see
src.agents.characterization_agent module docstring). This agent does not
diagnose a confirmed defect type -- it retrieves the knowledge-base
entries whose *process causes* most plausibly match the Characterization
Agent's heuristic Grad-CAM description, and asks the model to explain
likely process-related root causes *grounded in those entries*,
explicitly framed as inferred from activation pattern, not confirmed.

Only runs when `defect_characterization.applicable` is True -- same
skip-for-'ok' pattern as the Characterization Agent.
"""

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.agents.characterization_agent import DefectCharacterization
from src.agents.state import InspectionState
from src.knowledge import chroma_compat  # noqa: F401  (must precede `import chromadb`)
from src.knowledge.ingest import CHROMA_DB_DIR, COLLECTION_NAME

import anthropic
import chromadb

AGENT_NAME = "root_cause_agent"

REPO = Path(__file__).resolve().parents[2]
MODEL_ID = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024
TOP_K = 3

load_dotenv(REPO / ".env")

SYSTEM_PROMPT = """You are a casting-process quality engineer explaining the likely \
process-related root cause of a defect detected in an X-ray/visual inspection image \
of a die-cast part.

You are given:
1. A heuristic characterization of where the inspection model's Grad-CAM attention \
activated on the image (region size, position, confidence) -- this describes the \
model's attention pattern, not a confirmed defect diagnosis.
2. A small set of knowledge-base entries, each a known casting defect type and its \
typical process causes.

Ground your explanation strictly in the provided knowledge-base entries. Do not invent \
causes, defect types, or process details that are not present in the entries. If the \
characterization is ambiguous between entries, say so and explain the plausible \
alternatives rather than picking one arbitrarily. Be explicit that this is a \
possible-cause explanation inferred from an attention pattern, not a confirmed root \
cause -- a human inspector should verify. Keep the explanation to 2-4 sentences."""


@dataclass
class RootCause:
    """Root-Cause Agent output. `retrieved_entries` records exactly what
    context the explanation was grounded in, for traceability."""

    applicable: bool
    explanation: str
    retrieved_entries: List[Dict[str, Any]]
    model: Optional[str] = None


def _not_applicable() -> RootCause:
    return RootCause(
        applicable=False,
        explanation="Not applicable -- the Characterization Agent found no defect characterization to explain.",
        retrieved_entries=[],
    )


_collection = None


def _get_collection():
    """Load the persisted ChromaDB collection once per process and reuse it."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _build_query_text(characterization: DefectCharacterization) -> str:
    return (
        f"{characterization.category}: {characterization.description} "
        f"Region size: {characterization.region_size}. Position: {characterization.position}."
    )


def retrieve_knowledge(characterization: DefectCharacterization, n_results: int = TOP_K) -> List[Dict[str, Any]]:
    """Query the ChromaDB collection for the entries most relevant to this characterization."""
    collection = _get_collection()
    query_text = _build_query_text(characterization)

    result = collection.query(query_texts=[query_text], n_results=n_results)

    entries = []
    for metadata, distance in zip(result["metadatas"][0], result["distances"][0]):
        entries.append(
            {
                "defect_type": metadata["defect_type"],
                "process_causes": metadata["process_causes"],
                "distance": distance,
            }
        )
    return entries


def _build_user_message(characterization: DefectCharacterization, retrieved_entries: List[Dict[str, Any]]) -> str:
    knowledge_block = "\n\n".join(
        f"- {e['defect_type']}: {e['process_causes']}" for e in retrieved_entries
    )
    return f"""Grad-CAM characterization of the detected defect:
- Category: {characterization.category}
- Description: {characterization.description}
- Region size: {characterization.region_size}
- Position: {characterization.position}
- Model confidence tier: {characterization.confidence_tier}

Knowledge-base entries (most relevant, retrieved by similarity):
{knowledge_block}

Explain the likely process-related root cause(s) of this defect, grounded only in the \
knowledge-base entries above."""


_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def generate_explanation(characterization: DefectCharacterization, retrieved_entries: List[Dict[str, Any]]) -> str:
    client = _get_client()
    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_message(characterization, retrieved_entries)}],
    )
    return next(block.text for block in response.content if block.type == "text")


def explain_root_cause(characterization: DefectCharacterization) -> RootCause:
    """Retrieve relevant knowledge-base entries and generate a grounded explanation.

    Callers should only call this when `characterization.applicable` is
    True -- for a not-applicable characterization, use _not_applicable()
    (which the node function below does automatically).
    """
    retrieved_entries = retrieve_knowledge(characterization)
    explanation = generate_explanation(characterization, retrieved_entries)
    return RootCause(
        applicable=True,
        explanation=explanation,
        retrieved_entries=retrieved_entries,
        model=MODEL_ID,
    )


def root_cause_agent(state: InspectionState) -> Dict[str, Any]:
    """LangGraph node: RAG-generated root-cause explanation for a defective prediction.

    Only runs when the Characterization Agent's
    `defect_characterization.applicable` is True; otherwise returns a
    minimal not-applicable result. Returns a partial state update --
    LangGraph merges it into the full graph state -- rather than mutating
    `state` in place.
    """
    characterization = state.get("defect_characterization")
    if characterization is None or not characterization.applicable:
        result = _not_applicable()
    else:
        result = explain_root_cause(characterization)

    return {
        "root_cause": result,
        "agent_outputs": {AGENT_NAME: asdict(result)},
    }
