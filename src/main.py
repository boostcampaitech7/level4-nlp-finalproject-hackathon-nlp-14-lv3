from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inference import run_evaluation, run_inference, run_validation
from src.model import (
    EvaluationInput,
    EvaluationOutput,
    ServiceInput,
    ServiceOutput,
    ValidationInput,
    ValidationOutput,
)

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
