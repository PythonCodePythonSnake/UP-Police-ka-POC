"""LangGraph-powered dynamic conversation agent.

Coordinates extraction, sufficiency evaluation, knowledge retrieval (Pinecone boundary),
and dynamic conversational reply without relying on fixed question loops.
"""

import json
import logging
import re
from typing import Any, Optional, TypedDict

from pydantic import BaseModel, Field

from app.agent.prompts import GUIDANCE_PROMPT, INTAKE_PROMPT, SOURCES_UNAVAILABLE_NOTE
from app.agent.state_manager import (
    FIELD_DESCRIPTIONS,
    REQUIRED_FIELDS,
    ComplaintDetail,
    ComplaintState,
    apply_update,
    missing_required_fields,
)
from app.integrations.llm_client import (
    LLMConfigurationError,
    LLMError,
    LLMResponseError,
    LLMTimeoutError,
    generate_response,
    get_chat_model,
)
from app.schemas.chat import ChatMessage
from app.schemas.sources import RetrievalStatus, RetrievedSource
from app.services.retrieval_service import format_sources_for_prompt, retrieve_sources

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# LangGraph State Schema
# --------------------------------------------------------------------------- #
class AgentGraphState(TypedDict):
    session_id: str
    user_message: str
    messages: list[ChatMessage]
    complaint: ComplaintState
    missing_required: list[str]
    is_sufficient: bool
    need_retrieval: bool
    sources: list[RetrievedSource]
    sources_available: bool
    assistant_reply: str


# --------------------------------------------------------------------------- #
# Extraction Schemas
# --------------------------------------------------------------------------- #
class ExtractedDetail(BaseModel):
    value: str
    uncertain: bool = False


class ExtractionOutput(BaseModel):
    details: dict[str, Optional[ExtractedDetail]] = Field(
        default_factory=dict,
        description="Updated or newly extracted complaint details. Map field name to null if withdrawn.",
    )
    allegations_add: list[str] = Field(
        default_factory=list,
        description="Allegations made by complainant regarding wrongdoing.",
    )
    allegations_remove: list[str] = Field(
        default_factory=list,
        description="Allegations withdrawn or corrected by user.",
    )
    asks_for_procedure: bool = Field(
        default=False,
        description="True if the user asked about police procedures, circulars, legal sections, or guidance.",
    )


class SufficiencyOutput(BaseModel):
    is_sufficient: bool = Field(
        description="True if the collected information provides a clear, credible, and coherent picture of the incident to draft a report."
    )
    critical_missing_aspect: Optional[str] = Field(
        default=None,
        description="Brief note on what crucial aspect is missing or unclear (e.g. 'exact location not specified').",
    )


# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #
def _clean_markdown(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text


def _fields_context_prompt() -> str:
    return "\n".join(
        f"- {k}{' (required)' if k in REQUIRED_FIELDS else ''}: {v}"
        for k, v in FIELD_DESCRIPTIONS.items()
    )


# --------------------------------------------------------------------------- #
# Node 1: Dynamic Information Extraction
# --------------------------------------------------------------------------- #
async def extract_node(state: AgentGraphState) -> dict[str, Any]:
    """Extract facts, corrections, and detect if user asked procedural questions."""
    user_message = state["user_message"]
    current_complaint = state["complaint"]

    prompt = (
        "You are an expert complaint data extractor for a police assistance system.\n"
        "Examine the user's latest message and extract any factual details or corrections.\n"
        "Allowed complaint fields:\n"
        f"{_fields_context_prompt()}\n\n"
        "Current saved complaint details:\n"
        f"{current_complaint.model_dump_json(indent=2)}\n\n"
        "Rules:\n"
        "1. Extract ONLY facts explicitly stated by the user in this message. Never guess or infer.\n"
        "2. If user corrects/withdraws a detail, update it or set it to null.\n"
        "3. Set 'uncertain: true' if the user uses uncertain or approximate phrasing (e.g. 'around 9pm', 'maybe').\n"
        "4. 'allegations_add': wrongdoing claims formatted as 'The complainant states that...'.\n"
        "5. If user is merely greeting or chit-chatting, return empty details and allegations.\n"
        "6. Set 'asks_for_procedure: true' if the user asks about circulars, legal sections, rules, or police procedures.\n\n"
        "Return ONLY a valid JSON object matching this schema:\n"
        "{\n"
        '  "details": {"<field_name>": {"value": "<val>", "uncertain": false}},\n'
        '  "allegations_add": ["<claim>"],\n'
        '  "allegations_remove": ["<claim>"],\n'
        '  "asks_for_procedure": false\n'
        "}"
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"User message: {user_message}"},
    ]

    extraction = ExtractionOutput()
    try:
        model = get_chat_model(temperature=0.0)
        from langchain_core.messages import HumanMessage, SystemMessage
        lc_messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"User message: {user_message}"),
        ]
        response = await model.ainvoke(lc_messages)
        clean_content = _clean_markdown(response.content if isinstance(response.content, str) else str(response.content))
        data = json.loads(clean_content)
        extraction = ExtractionOutput.model_validate(data)
    except Exception as exc:
        logger.warning("LangChain structured extraction fallback to direct LLM call: %s", exc)
        # Fallback to direct generate_response if langchain provider fails or during mock tests
        try:
            from app.integrations.llm_client import LLMMessage
            raw = await generate_response(
                [LLMMessage("system", prompt), LLMMessage("user", f"User message: {user_message}")],
                json_mode=True,
                temperature=0.0,
            )
            data = json.loads(_clean_markdown(raw))
            extraction = ExtractionOutput.model_validate(data)
        except Exception as e:
            logger.error("Extraction error: %s", e)

    # Merge into complaint state
    updated_complaint = current_complaint.model_copy(deep=True)
    converted_details = {}
    for k, v in extraction.details.items():
        if v is None:
            converted_details[k] = None
        else:
            converted_details[k] = ComplaintDetail(value=v.value, uncertain=v.uncertain)

    apply_update(
        updated_complaint,
        converted_details,
        extraction.allegations_add,
        extraction.allegations_remove,
    )

    missing = missing_required_fields(updated_complaint)
    return {
        "complaint": updated_complaint,
        "missing_required": missing,
        "need_retrieval": extraction.asks_for_procedure,
    }


# --------------------------------------------------------------------------- #
# Node 2: Dynamic Sufficiency Evaluation
# --------------------------------------------------------------------------- #
async def evaluate_sufficiency_node(state: AgentGraphState) -> dict[str, Any]:
    """Dynamically evaluate whether details collected are sufficient for a complaint report."""
    complaint = state["complaint"]
    missing = state["missing_required"]

    # Basic hard rule: if required fields are missing, not yet sufficient
    if missing:
        return {"is_sufficient": False}

    # Evaluate completeness & clarity of the report using LLM
    prompt = (
        "You are a senior police intake officer evaluating a complaint record for sufficiency.\n"
        "Check if the collected complaint details contain enough clear, actionable facts to prepare an official draft.\n"
        f"Complaint Record:\n{complaint.model_dump_json(indent=2)}\n\n"
        "Evaluate whether the narrative makes sense and has key elements (who, when, where, what happened, contact).\n"
        "Return ONLY a JSON object:\n"
        '{"is_sufficient": true, "critical_missing_aspect": null}'
    )

    try:
        from app.integrations.llm_client import LLMMessage
        raw = await generate_response(
            [LLMMessage("system", prompt), LLMMessage("user", "Assess sufficiency.")],
            json_mode=True,
            temperature=0.0,
        )
        res = json.loads(_clean_markdown(raw))
        is_sufficient = bool(res.get("is_sufficient", True))
    except Exception:
        # If evaluator fails or in testing, fall back to whether missing_required is empty
        is_sufficient = len(missing) == 0

    return {"is_sufficient": is_sufficient}


# --------------------------------------------------------------------------- #
# Node 3: Knowledge Retrieval Boundary (Pinecone Plug)
# --------------------------------------------------------------------------- #
async def retrieve_node(state: AgentGraphState) -> dict[str, Any]:
    """Query Pinecone if the user asked procedural guidance or if retrieval is warranted."""
    user_message = state["user_message"]
    complaint = state["complaint"]

    retrieval_resp = await retrieve_sources(
        query=user_message,
        complaint=complaint,
        top_k=3,
    )

    sources_available = (retrieval_resp.status == RetrievalStatus.SUCCESS and bool(retrieval_resp.sources))
    return {
        "sources": retrieval_resp.sources,
        "sources_available": sources_available,
    }


# --------------------------------------------------------------------------- #
# Node 4: Dynamic Conversational Responder
# --------------------------------------------------------------------------- #
async def respond_node(state: AgentGraphState) -> dict[str, Any]:
    """Generate a dynamic, adaptive conversational response."""
    user_message = state["user_message"]
    complaint = state["complaint"]
    missing = state["missing_required"]
    is_sufficient = state["is_sufficient"]
    sources = state["sources"]
    sources_available = state["sources_available"]
    history = state["messages"]

    # Source prompt context
    if sources_available:
        from app.schemas.sources import RetrievalResponse, RetrievalStatus
        mock_resp = RetrievalResponse(status=RetrievalStatus.SUCCESS, sources=sources)
        sources_text = format_sources_for_prompt(mock_resp) or ""
    else:
        sources_text = SOURCES_UNAVAILABLE_NOTE

    # Dynamic instructions based on state
    if is_sufficient:
        strategy_instruction = (
            "All essential details have been collected and the report is sufficiently detailed.\n"
            "- Summarize the key facts recorded in short, respectful bullet points.\n"
            "- Clearly note that this is unverified and not yet officially filed.\n"
            "- Ask the user if anything is inaccurate, if they'd like to correct/add anything, "
            "or if they are ready to review the complaint draft."
        )
    else:
        missing_names = ", ".join(missing) if missing else "clarifying details"
        strategy_instruction = (
            f"The complaint is not yet complete. Missing or needed aspects: {missing_names}.\n"
            "- Acknowledge what the user just shared.\n"
            "- Ask at most 1 or 2 focused, natural follow-up questions to gather the missing details.\n"
            "- Prioritize the most critical missing info first.\n"
            "- NEVER ask for information that is already provided in the complaint details."
        )

    system_prompt = (
        f"{INTAKE_PROMPT}\n\n"
        f"{GUIDANCE_PROMPT}\n\n"
        f"SOURCES INFORMATION:\n{sources_text}\n\n"
        f"CURRENT RECORDED COMPLAINT:\n{complaint.model_dump_json(indent=2)}\n\n"
        f"STRATEGY:\n{strategy_instruction}\n\n"
        "Reply with a conversational, natural assistant response in plain text. "
        "Do not include code fences or raw JSON."
    )

    from app.integrations.llm_client import LLMMessage
    llm_messages = [LLMMessage("system", system_prompt)]
    for m in history:
        llm_messages.append(LLMMessage(m.role, m.content))
    llm_messages.append(LLMMessage("user", user_message))

    reply_text = await generate_response(llm_messages, json_mode=False, temperature=0.3)
    return {"assistant_reply": reply_text.strip()}


# --------------------------------------------------------------------------- #
# Routing & Graph Compilation
# --------------------------------------------------------------------------- #
def route_after_evaluation(state: AgentGraphState) -> str:
    if state.get("need_retrieval", False):
        return "retrieve"
    return "respond"


def build_complaint_graph():
    """Build and compile the LangGraph workflow."""
    from langgraph.graph import END, StateGraph

    workflow = StateGraph(AgentGraphState)

    workflow.add_node("extract", extract_node)
    workflow.add_node("evaluate", evaluate_sufficiency_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("respond", respond_node)

    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "retrieve": "retrieve",
            "respond": "respond",
        },
    )
    workflow.add_edge("retrieve", "respond")
    workflow.add_edge("respond", END)

    return workflow.compile()
