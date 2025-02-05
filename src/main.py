from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import run_evaluation, run_inference, run_validation


# 요청 데이터 모델 정의
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


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/inference", response_model=ServiceOutput)
async def inference(request: ServiceInput):
    text = request.messages[0].text
    print(f"Message: f{text}")  # For debugging
    response = run_inference(text)
    return {"text": response}


@app.post("/evaluation", response_model=EvaluationOutput)
async def evaluation(request: EvaluationInput):
    query = request.query
    return run_evaluation(query)


@app.post("/validation", response_model=ValidationOutput)
async def validation(request: ValidationInput):
    train_test_ratio = request.train_test_ratio
    return run_validation(train_test_ratio)


# FastAPI 서버 실행 명령어
# uvicorn main:app --port 8000 --reload
# cloudflared tunnel --url http://localhost:8000
