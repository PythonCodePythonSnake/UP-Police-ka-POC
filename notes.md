# LangGraph Migration & Agent Redesign Notes

## Objective
Migrate the conversational complaint filing assistant from a single-turn fixed prompt loop in `orchestrator.py` to a dynamic, adaptive agent using **LangGraph / LangChain**, while:
1. Keeping all FastAPI API endpoints, paths, status codes, and input/output schemas strictly unmodified.
2. Keeping Pinecone modular, un-mocked, and ready to plug via `retrieval_service.py`.
3. Supporting both **OpenAI** and **Google Gemini** models via `llm_client.py`.
4. Deferring running tests until the user sets up the virtual environment (`venv`).

---

## 1. Endpoints & Schemas (Strictly Preserved)
- `GET /api/health` -> `HealthResponse(app, llm, retrieval_configured)`
- `POST /api/sessions` -> `SessionCreateResponse`
- `GET /api/sessions/{session_id}` -> `SessionHistoryResponse`
- `DELETE /api/sessions/{session_id}` -> 204 No Content
- `POST /api/chat/{session_id}` -> Request: `ChatRequest(message: str)`, Response: `ChatResponse(session, reply, complaint, missing_required_fields, required_details_collected, sources_available)`
- `GET /api/complaints/{session_id}` -> `ComplaintResponse`
- `PATCH /api/complaints/{session_id}` -> Request: `ComplaintUpdateRequest`, Response: `ComplaintResponse`
- `POST /api/complaints/{session_id}/draft` -> `ComplaintDraft`

---

## 2. Dependencies to Add to `requirements.txt`
In addition to `pydantic`, `pydantic_settings`, `httpx`, `fastapi[standard]`, `pytest`:
- `langchain-core`
- `langgraph`
- `langchain-openai`
- `langchain-google-genai`

---

## 3. Plan for `app/integrations/llm_client.py`
Refactor / extend `llm_client.py` to:
- Retain the custom error hierarchy: `LLMError`, `LLMConfigurationError`, `LLMTimeoutError`, `LLMProviderError`, `LLMResponseError`.
- Provide a clean factory function `get_chat_model(temperature: float = 0.0, **kwargs)` returning a LangChain `BaseChatModel`:
  - `ChatOpenAI(model=settings.llm_model, api_key=..., timeout=settings.llm_timeout_seconds)` for provider `"openai"`.
  - `ChatGoogleGenerativeAI(model=settings.llm_model, google_api_key=..., timeout=settings.llm_timeout_seconds)` for provider `"gemini"`.
- Support structured outputs via `.with_structured_output(...)` or fallback schema parsing.
- Retain `generate_response(...)` (or make it wrap the model / keep compatibility) so `draft_service.py` and existing tests continue to work or can be invoked smoothly.

---

## 4. Plan for `app/agent/graph.py` (LangGraph StateGraph)
### State Schema: `ComplaintAgentState`
```python
class ComplaintAgentState(TypedDict):
    session_id: str
    messages: list[ChatMessage]  # recent conversation history
    user_message: str
    complaint_state: ComplaintState
    missing_required_fields: list[str]
    is_sufficient: bool          # Evaluated readiness for a report draft
    need_retrieval: bool         # Whether legal / circular query is detected
    sources: list[RetrievedSource]
    sources_available: bool
    assistant_reply: str
```

### Graph Nodes:
1. **`extract_node`**:
   - Analyzes user utterance against existing `ComplaintState`.
   - Extracts updated/corrected fields (with `uncertain` flag) and added/removed allegations.
   - Merges updates into `complaint_state` using `state_manager.apply_update`.
2. **`evaluate_sufficiency_node`**:
   - Assesses if current details constitute a clear, coherent, and actionable report.
   - Determines missing required fields (`state_manager.missing_required_fields`) and checks if procedural guidance is requested.
3. **`retrieval_node`**:
   - If user asks about rules/procedures or complaint category guidance:
     Calls `retrieval_service.retrieve_sources(query, complaint)`.
     Sets `sources_available` according to Pinecone status.
4. **`respond_node`**:
   - If sufficient: summarizes user-reported facts politely and asks if anything needs correcting before drafting.
   - If incomplete: asks 1-2 prioritized, non-repetitive, context-aware follow-up questions for the missing details.
   - Respects disclaimer rules (no false claims of filing or fabricated circulars).

---

## 5. Plan for `app/agent/orchestrator.py`
- Expose the exact signature:
  `async def handle_user_message(session: SessionData, user_message: str, service: SessionService) -> TurnResult`
- Invoke the compiled LangGraph workflow.
- Persist user message, assistant reply, and updated `ComplaintState` to `session_service`.
- Keep `session_lock(session_id)` for concurrency protection.
