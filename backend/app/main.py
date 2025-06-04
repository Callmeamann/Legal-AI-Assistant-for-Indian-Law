# adaptive_rag_ui/app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware
from .schemas import QuestionRequest, AnswerResponse
from .rag_logic import get_answer_from_rag, initialize_rag_components

app = FastAPI(title="Adaptive RAG Legal Advisor API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for simplicity, restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.on_event("startup")
async def startup_event():
    """
    Initialize RAG components when the application starts.
    """
    print("Initializing RAG components...")
    initialize_rag_components()
    print("RAG components initialized.")

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Endpoint to receive a question and return an answer from the RAG system.
    """
    try:
        print(f"Received question: {request.question}")
        answer, sources = await get_answer_from_rag(request.question)
        if answer is None:
            raise HTTPException(status_code=500, detail="Failed to get an answer from the RAG system.")
        print(f"Generated answer: {answer}")
        return AnswerResponse(answer=answer, sources=sources)
    except Exception as e:
        print(f"Error during /ask endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Adaptive RAG Legal Advisor API"}

if __name__ == "__main__":
    import uvicorn
    # This is for local development. For production, use a proper ASGI server like Gunicorn.
    uvicorn.run(app, host="0.0.0.0", port=8000)
