from pydantic import BaseModel
from typing import List, Optional

class QuestionRequest(BaseModel):
    question: str
    chat_history: Optional[List[dict]] = []

class AnswerResponse(BaseModel):
    answer: str
    sources: Optional[List[dict]] = []
    metadata: Optional[dict] = {} 