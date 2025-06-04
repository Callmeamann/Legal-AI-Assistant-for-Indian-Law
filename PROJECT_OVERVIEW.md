# Legal AI Assistant – Project Overview & Presentation Script

---

## 1. Mission
Deliver a professional, **Indian-law–focused** AI assistant that answers legal questions using the new Bharatiya Nyaya Sanhita (BNS), related legislation, and Supreme Court precedents.  
Built with a Retrieval-Augmented-Generation (RAG) backend and a modern chat-style frontend.

---

## 2. High-Level Architecture
```
User ↔ Next.js Frontend ↔ FastAPI Backend ↔ LangGraph-powered RAG System ↔
         ├─ Supreme-Court Vector DB (Chroma)          
         └─ Tavily Web Search API
```
1. **Frontend (Next.js 15 + Tailwind)** – chat UI, markdown rendering, avatars, loading states, clear-chat, mobile responsive.
2. **Backend (FastAPI)** – single `/ask` endpoint; handles CORS and calls RAG pipeline.
3. **RAG Pipeline (LangGraph)**  
   • Embeddings: Sentence-Transformers  
   • Vector DB: Chroma (persisted in `backend/db_data/sc-chroma`)  
   • LLM: Llama-3-70B via Groq API  
   • Web search: Tavily API  
   • Graph nodes: routing → retrieval → grading → generation → grading.

---

## 3. Backend Walkthrough

| Stage | Key Code | Purpose |
|-------|----------|---------|
| **Initialization** | `initialize_rag_components()` | Loads models, vectorstore, tools, prompts, and compiles the LangGraph workflow. |
| **Routing** | `route_question_edge` | LLM decides whether to use SC vectorstore, web search, or both. |
| **Retrieval Nodes** | `retrieve_sc_node`, `retrieve_web_search_node`, `retrieve_combined_sources_node` | Pull relevant documents (k = 5) or search results (k = 3). |
| **Document Grading** | `grade_documents_node` | LLM grades each doc for relevance (truncates to 2 k chars & safe-try). |
| **Query Rewrite** | `transform_query_node` | Improves the question if no good docs found. |
| **Answer Generation** | `generate_answer_node` + RAG prompt | Produces structured legal answer (summary, provisions, analysis, final). |
| **Answer Grading** | `grade_generation_edge` | Checks hallucination + usefulness; may regenerate or rewrite. |
| **Safety Guard** | `MAX_GENERATION_ATTEMPTS = 5` | Stops infinite loops and returns fallback *"Sorry, we could not understand your query…"*. |

Endpoint: `POST /ask`  
```json
{
  "question": "What is the procedure for filing an FIR?",
  "chat_history": []   // (reserved for future multi-turn use)
}
```
Returns: `answer` (markdown) + `sources` (URL metadata).

---

## 4. Frontend Walkthrough

Component: `components/ChatInterface.tsx`

1. **State**: `messages`, `input`, `isLoading`, `error`.
2. **UX perks**: avatars, markdown via `react-markdown` + `remark-gfm`, thumbs-up/down placeholders, clear-chat, loading spinner.
3. **Flow**:
   ```tsx
   await fetch("http://127.0.0.1:8000/ask", {...})
   → push assistant message with answer & sources
   ```
4. **Responsive layout**: flex column → mobile stack; Tailwind `prose` for typography.

---

## 5. End-to-End Flow
1. User types a legal query → clicks **Send**.
2. Frontend POSTs to `/ask`.
3. Backend's LangGraph pipeline:
   1. Router chooses `sc_vectorstore` / `web_search`.
   2. Retrieves docs → grades relevance.
   3. If no good docs, question rewritten → new retrieval.
   4. RAG prompt + docs → draft answer.
   5. Hallucination & usefulness graded.
   6. If bad, regenerate or rewrite (max 5 loops).
4. Final answer & sources returned.
5. Frontend renders markdown-styled answer, shows clickable sources.

---

## 6. Running Locally (Demo Setup)
```bash
# 1) Backend
cd backend
python -m venv venv && venv\Scripts\activate      # Windows example
pip install -r requirements.txt
set GROQ_API_KEY=...        && set TAVILY_API_KEY=...
uvicorn app.main:app --reload

# 2) Frontend (new shell)
cd nextjs_frontend
npm install
npm run dev
```
Navigate to <http://localhost:3000>

---

## 7. Live Presentation Script (≈ 4 min)
1. **Opening (30 s)**  
   "Good day judges. I present the *Legal AI Assistant*, designed to democratize access to up-to-date Indian legal knowledge."
2. **Problem (20 s)**  
   "With BNS replacing IPC, practitioners need quick, reliable answers—existing search tools are slow and fragmented."
3. **Solution Overview (40 s)**  
   Show architecture slide / diagram above.  Emphasise RAG to ground answers in law + cases.
4. **Live Demo (90 s)**  
   • Ask: *"Explain the procedure for filing an FIR under BNSS."*  
   • Highlight bold / italic markdown, structured answer sections, source links.  
   • Clear chat, ask *"What was the SC view on Triple Talaq?"*  ‑ show Supreme-Court citations.
5. **Safety & Quality (30 s)**  
   Explain grading nodes, hallucination guard, retry cap.
6. **Tech Highlights (20 s)**  
   Groq Llama-3-70B speed, Chroma vectorstore, LangGraph orchestration.
7. **Closing (30 s)**  
   "Scalable to other jurisdictions, integrates new caselaw instantly via web search. Ready to assist lawyers, journalists, and citizens alike."

---

## 8. Future Work
* Multi-turn memory using chat_history.  
* Feedback buttons hooked to a scoring API for continual improvement.  
* Deploy on GPU-backed cloud with authentication & usage analytics.

---

*Prepared by Aman Gusain aka Callmeamann* 
