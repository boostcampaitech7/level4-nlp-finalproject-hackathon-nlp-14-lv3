from typing import List, Optional

from pydantic import BaseModel


class MessageContent(BaseModel):
    role: Optional[str] = "ai"
    text: Optional[str] = None
    files: Optional[List[str]] = None
    html: Optional[str] = None


class ServiceInput(BaseModel):
    messages: List[MessageContent]


class ServiceOutput(BaseModel):
    text: str


class EvaluationInput(BaseModel):
    query: str


class EvaluationOutput(BaseModel):
    context: list[str]
    answer: str


class ValidationInput(BaseModel):
    train_test_ratio: str


class ValidationOutput(BaseModel):
    question: str
    context: list[str]
    answer: str


class GEvalResult(BaseModel):
    query: str
    retrieval_contexts: List[str]
    generated_answer: str
    expected_answer: str
    retrieval_score: List[int]  # out of 20
    generation_score: List[int]  # out of 30
    total_score: int  # out of 50
