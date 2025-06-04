# adaptive_rag_ui/app/rag_logic.py
import os
from typing import List, Literal, Tuple, Optional, TypedDict
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph, START
# from typing_extensions import TypedDict # Already imported above

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
# IMPORTANT: Update this path to where your ChromaDB is located
# db_data is in the backend directory
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db_data", "sc-chroma")
LLM_MODEL_NAME = "llama3-70b-8192" # Using the 70b model as in the notebook
MAX_GENERATION_ATTEMPTS = 5  # Limit retries to prevent infinite hallucination loops
MAX_DOC_CHARS_FOR_GRADING = 2000  # prevent oversize prompt errors

# --- Global RAG Components (will be initialized by initialize_rag_components) ---
llm = None
embedding_model = None
sc_vectorstore = None
sc_retriever = None
web_search_tool = None
question_router = None
retrieval_grader = None
rag_chain = None
hallucination_grader = None
answer_grader = None
question_rewriter = None
compiled_rag_app = None

# --- Pydantic Models for Graders and Routers (from notebook) ---
class RouteQuery(BaseModel):
    """Choose the best source(s) for answering the legal question."""
    datasources: List[Literal["sc_vectorstore", "web_search"]] = Field(
        ..., description="List of sources to retrieve relevant information from"
    )

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

# --- LangGraph State ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        original_question: To keep track of the initial question for re-writing scenarios
    """
    question: str
    generation: str
    documents: List[Document]
    original_question: str

# --- Initialization Function ---
def initialize_rag_components():
    global llm, embedding_model, sc_vectorstore, sc_retriever, web_search_tool
    global question_router, retrieval_grader, rag_chain, hallucination_grader
    global answer_grader, question_rewriter, compiled_rag_app

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY must be set in .env file.")
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY must be set in .env file.")

    # Initialize LLM
    llm = ChatGroq(model=LLM_MODEL_NAME, temperature=0, groq_api_key=GROQ_API_KEY)

    # Initialize Embedding Model
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Initialize Vector Store and Retriever
    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(
            f"Chroma DB path not found: {CHROMA_DB_PATH}. "
            "Please ensure your 'sc-chroma' directory is correctly placed in 'db_data' and the path is updated."
        )
    sc_vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding_model,
        collection_name="supreme-court-db" # As per notebook
    )
    sc_retriever = sc_vectorstore.as_retriever(search_kwargs={"k": 5})

    # Initialize Web Search Tool
    web_search_tool = TavilySearchResults(k=3, tavily_api_key=TAVILY_API_KEY) # k=3 as in notebook

    # Initialize Question Router
    structured_llm_router = llm.with_structured_output(RouteQuery)
    system_router = """
You are an expert at routing legal questions.
The available sources are:
- sc_vectorstore: Supreme Court cases and precedents.
- web_search: Dynamic online search for current events or extra context.
Rules:
- If the user asks about court cases, judgments, or case-based interpretations, use 'sc_vectorstore'.
- If the query requires recent information or public commentary, use 'web_search'.
- You may return both if the question would benefit from both sources.
Respond with a list of one or more datasources.
"""
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", system_router),
        ("human", "{question}"),
    ])
    question_router = route_prompt | structured_llm_router

    # Initialize Retrieval Grader
    structured_llm_retrieval_grader = llm.with_structured_output(GradeDocuments)
    system_retrieval_grader = """You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system_retrieval_grader),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    retrieval_grader = grade_prompt | structured_llm_retrieval_grader

    # Initialize RAG Chain (Generator)
    system_rag = """You are a legal assistant answering queries based on Indian laws. Use the most current legal framework in force, including:
- Bharatiya Nyaya Sanhita (BNS)
- Bharatiya Nagarik Suraksha Sanhita (BNSS)
- Bharatiya Sakshya Adhiniyam (BSA)
Forget referencing and using the repealed Indian Penal Code (IPC), Criminal Procedure Code (CrPC), and Indian Evidence Act for the new laws.
In addition, you may reference any applicable Indian legislation that remains in force, such as:
- The Motor Vehicles Act, 1988
- The Information Technology Act, 2000
- The Companies Act, 2013
- The Contract Act, 1872
- The Juvenile Justice (Care and Protection of Children) Act, 2015
- Any relevant Central or State laws still valid
Use relevant Supreme Court or High Court judgments when applicable for authoritative reference.
Respond in a clear, structured format:
1. Summary
2. Relevant Legal Provisions (mention section numbers and case law if available)
3. Legal Analysis or Reasoning
4. Final Answer
If the question lacks sufficient context or legal basis for a definitive answer, clearly state that.
Be precise, formal, and helpful.
"""
    rag_prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_rag),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    rag_chain = rag_prompt_template | llm | StrOutputParser()

    # Initialize Hallucination Grader
    structured_llm_hallucination_grader = llm.with_structured_output(GradeHallucinations)
    system_hallucination = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", system_hallucination),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ])
    hallucination_grader = hallucination_prompt | structured_llm_hallucination_grader

    # Initialize Answer Grader
    structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)
    system_answer = """You are a grader assessing whether an answer addresses / resolves a question.
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", system_answer),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ])
    answer_grader = answer_prompt | structured_llm_answer_grader

    # Initialize Question Re-writer
    system_rewrite = """You a question re-writer that converts an input question to a better version that is optimized
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
     Also only output the rewritten question, without any extra text or explanation."""
    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system_rewrite),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ])
    question_rewriter = re_write_prompt | llm | StrOutputParser()

    # --- LangGraph Workflow Definition ---
    workflow = StateGraph(GraphState)

    # Define nodes
    workflow.add_node("retrieve_sc", retrieve_sc_node)
    workflow.add_node("retrieve_web_search", retrieve_web_search_node)
    workflow.add_node("retrieve_combined_sources", retrieve_combined_sources_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("transform_query", transform_query_node)

    # Define edges
    workflow.add_conditional_edges(
        START,
        route_question_edge,
        {
            "sc_vectorstore": "retrieve_sc",
            "web_search": "retrieve_web_search",
            "both": "retrieve_combined_sources", # Route to combined if both are needed
        }
    )
    workflow.add_edge("retrieve_sc", "grade_documents")
    workflow.add_edge("retrieve_web_search", "grade_documents")
    workflow.add_edge("retrieve_combined_sources", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate_edge,
        {
            "transform_query": "transform_query",
            "generate": "generate_answer",
        }
    )
    # After transforming query, always try SC retrieval first as per notebook logic.
    # If web search is needed again, the router should handle it in a subsequent cycle if the graph is designed for cycles.
    # For simplicity here, we'll go back to SC retrieval.
    workflow.add_edge("transform_query", "retrieve_sc")

    workflow.add_conditional_edges(
        "generate_answer",
        grade_generation_edge,
        {
            "not_supported": "generate_answer",  # Retry generation if not grounded
            "useful": END, # End if useful
            "not_useful": "transform_query", # If not useful, transform original question and retry
        }
    )
    compiled_rag_app = workflow.compile()


# --- LangGraph Node Functions ---
def format_docs_for_context(documents: List[Document]) -> str:
    """Helper function to format a list of Document objects into a single string for context."""
    return "\n\n".join(doc.page_content for doc in documents)

async def retrieve_sc_node(state: GraphState) -> GraphState:
    """Retrieves documents from the Supreme Court vectorstore."""
    print("---NODE: RETRIEVE FROM SC VECTORSTORE---")
    question = state["question"]
    # Ensure retriever is initialized
    if sc_retriever is None:
        print("Error: SC Retriever not initialized.")
        return {**state, "documents": []}
    documents = await sc_retriever.aget_relevant_documents(question)
    print(f"Retrieved {len(documents)} documents from SC.")
    return {**state, "documents": documents}

async def retrieve_web_search_node(state: GraphState) -> GraphState:
    """Retrieves documents using web search."""
    print("---NODE: PERFORMING WEB SEARCH---")
    question = state["question"]
    if web_search_tool is None:
        print("Error: Web Search Tool not initialized.")
        return {**state, "documents": []}
    # TavilySearchResults returns a list of dicts, convert to Document objects
    search_results = await web_search_tool.ainvoke({"query": question})
    documents = [Document(page_content=d["content"], metadata={"source": d.get("url", "web_search")}) for d in search_results]
    print(f"Found {len(documents)} web search results.")
    return {**state, "documents": documents}

async def retrieve_combined_sources_node(state: GraphState) -> GraphState:
    """Combines retrieval from Supreme Court vectorstore and web search."""
    print("---NODE: PERFORMING COMBINED RETRIEVAL (SC + WEB)---")
    question = state["question"]

    sc_docs = []
    if sc_retriever:
        sc_docs_result = await sc_retriever.aget_relevant_documents(question)
        sc_docs.extend(sc_docs_result)
    print(f"Retrieved {len(sc_docs)} documents from SC for combined search.")

    web_docs = []
    if web_search_tool:
        search_results = await web_search_tool.ainvoke({"query": question})
        web_docs.extend([Document(page_content=d["content"], metadata={"source": d.get("url", "web_search")}) for d in search_results])
    print(f"Found {len(web_docs)} web search results for combined search.")

    combined_docs = sc_docs + web_docs
    print(f"Total combined documents: {len(combined_docs)}")
    return {**state, "documents": combined_docs}

async def grade_documents_node(state: GraphState) -> GraphState:
    """Grades the relevance of retrieved documents."""
    print("---NODE: GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]
    if retrieval_grader is None:
        print("Error: Retrieval Grader not initialized.")
        return {**state, "documents": []}

    filtered_docs = []
    for doc in documents:
        # Ensure doc.page_content is not None
        if doc.page_content is None:
            print(f"Warning: Document with None page_content found: {doc.metadata}")
            continue
        # Truncate document to avoid exceeding model or API limits
        text_for_grade = doc.page_content[:MAX_DOC_CHARS_FOR_GRADING]
        try:
            grade = await retrieval_grader.ainvoke({"question": question, "document": text_for_grade})
        except Exception as e:
            print(f"Grader error on document (source: {doc.metadata.get('source', 'N/A')}): {str(e)}. Skipping document.")
            continue
        if grade.binary_score.lower() == "yes":
            print(f"---GRADE: DOCUMENT RELEVANT (Source: {doc.metadata.get('source', 'N/A')})---")
            filtered_docs.append(doc)
        else:
            print(f"---GRADE: DOCUMENT NOT RELEVANT (Source: {doc.metadata.get('source', 'N/A')})---")
    return {**state, "documents": filtered_docs}

async def generate_answer_node(state: GraphState) -> GraphState:
    """Generates an answer using the RAG chain."""
    print("---NODE: GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"] # These are now List[Document]
    if rag_chain is None:
        print("Error: RAG Chain not initialized.")
        return {**state, "generation": "Error: RAG chain not available."}

    # Format documents for context
    context_str = format_docs_for_context(documents)
    generation = await rag_chain.ainvoke({"context": context_str, "question": question})
    return {**state, "generation": generation}

async def transform_query_node(state: GraphState) -> GraphState:
    """Transforms the query to a better version for retrieval."""
    print("---NODE: TRANSFORM QUERY---")
    # Use original_question if available and this is a retry, otherwise use current question
    question_to_rewrite = state.get("original_question", state["question"])

    if question_rewriter is None:
        print("Error: Question Rewriter not initialized.")
        return {**state, "question": question_to_rewrite} # Return original if rewriter fails

    better_question = await question_rewriter.ainvoke({"question": question_to_rewrite})
    print(f"---TRANSFORMED QUERY to: {better_question}---")
    return {**state, "question": better_question} # Keep original_question as is

# --- LangGraph Edge Functions ---
async def route_question_edge(state: GraphState) -> Literal["sc_vectorstore", "web_search", "both"]:
    """Routes the question to the appropriate data source."""
    print("---EDGE: ROUTE QUESTION---")
    question = state["question"]
    if question_router is None:
        print("Error: Question Router not initialized. Defaulting to web_search.")
        return "web_search" # Default route

    source_decision = await question_router.ainvoke({"question": question})
    print(f"Router decision: {source_decision.datasources}")
    if "sc_vectorstore" in source_decision.datasources and "web_search" in source_decision.datasources:
        print("---ROUTE: BOTH SC VECTORSTORE AND WEB SEARCH---")
        return "both"
    elif "sc_vectorstore" in source_decision.datasources:
        print("---ROUTE: SC VECTORSTORE---")
        return "sc_vectorstore"
    else: # Default to web_search if only web_search or empty
        print("---ROUTE: WEB SEARCH---")
        return "web_search"

async def decide_to_generate_edge(state: GraphState) -> Literal["transform_query", "generate"]:
    """Decides whether to generate an answer or transform the query."""
    print("---EDGE: DECIDE TO GENERATE---")
    filtered_documents = state["documents"]
    if not filtered_documents:
        print("---DECISION: NO RELEVANT DOCUMENTS, TRANSFORM QUERY---")
        return "transform_query"
    else:
        print("---DECISION: RELEVANT DOCUMENTS FOUND, GENERATE ANSWER---")
        return "generate"

async def grade_generation_edge(state: GraphState) -> Literal["not_supported", "useful", "not_useful"]:
    """Grades the generated answer for hallucination and usefulness."""
    print("---EDGE: GRADE GENERATION---")
    question = state.get("original_question", state["question"]) # Grade against original question
    documents = state["documents"]
    generation = state["generation"]

    if hallucination_grader is None or answer_grader is None:
        print("Error: Graders not initialized. Defaulting to 'useful'.")
        return "useful"

    # Format documents for hallucination grader
    docs_str = format_docs_for_context(documents)

    hallucination_score = await hallucination_grader.ainvoke({"documents": docs_str, "generation": generation})
    if hallucination_score.binary_score.lower() == "no":
        print("---GRADE: GENERATION NOT GROUNDED (HALLUCINATION) - RETRY GENERATION---")
        return "not_supported" # Retry generation

    print("---GRADE: GENERATION IS GROUNDED---")
    answer_score = await answer_grader.ainvoke({"question": question, "generation": generation})
    if answer_score.binary_score.lower() == "yes":
        print("---GRADE: GENERATION IS USEFUL---")
        return "useful"
    else:
        print("---GRADE: GENERATION NOT USEFUL - TRANSFORM QUERY---")
        return "not_useful"


# --- Main function to get answer ---
async def get_answer_from_rag(user_question: str) -> Tuple[Optional[str], List[dict]]:
    """
    Gets an answer from the RAG system for a given question.
    Returns the answer and a list of source documents' content.
    """
    if compiled_rag_app is None:
        print("Error: RAG App not compiled/initialized.")
        return "Sorry, we could not understand your query. Please try again.", []

    initial_state = GraphState(
        question=user_question,
        generation="",
        documents=[],
        original_question=user_question # Store the original question
    )
    final_state = None
    generation_attempts = 0  # Track how many times the LLM has attempted to generate an answer
    try:
        print(f"\nInvoking RAG app with question: {user_question}")
        async for event in compiled_rag_app.astream(initial_state):
            for node_name, node_output in event.items():
                # Detect each time the `generate_answer` node runs
                if node_name == "generate_answer":
                    generation_attempts += 1
                    if generation_attempts > MAX_GENERATION_ATTEMPTS:
                        # Exceeded retry limit – abort and return fallback response
                        print("Maximum generation attempts exceeded – returning fallback message.")
                        return (
                            "Sorry, we could not understand your query. Please try again.",
                            []
                        )

                if isinstance(node_output, dict):
                    final_state = node_output

        if final_state and final_state.get("generation"):
            answer = final_state["generation"]
            # Format sources as dictionaries with content and metadata
            sources = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in final_state.get("documents", [])
                if doc.page_content
            ]
            print(f"Final answer: {answer}")
            print(f"Sources used: {len(sources)}")
            return answer, sources
        else:
            print("RAG app did not produce a final generation.")
            return "Sorry, we could not understand your query. Please try again.", []

    except Exception as e:
        print(f"Error running RAG app: {str(e)}")
        return f"An error occurred: {str(e)}", []

if __name__ == '__main__':
    # This is for testing the rag_logic.py directly
    import asyncio

    async def main_test():
        initialize_rag_components() # Initialize all components
        # Test question
        # test_q = "What is the current procedure for filing an FIR?"
        test_q = "What did the Supreme Court rule on triple talaq and how is it viewed today?"
        answer, sources = await get_answer_from_rag(test_q)
        print("\n--- TEST RESULT ---")
        print(f"Question: {test_q}")
        print(f"Answer: {answer}")
        if sources:
            print("\nSources:")
            for i, src_content in enumerate(sources):
                print(f"Source {i+1}: {src_content['content'][:200]}...") # Print first 200 chars
        else:
            print("No sources provided.")

    asyncio.run(main_test())
