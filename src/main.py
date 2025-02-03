from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional
from inference import run_inference, retrieve_contexts, generate_answer  # QA (inference) 함수 불러오기


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


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     yield
# # FastAPI 앱 생성
# app = FastAPI(lifespan=lifespan)

app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 허용할 출처 추가 "http://localhost:8080", "http://localhost:5173"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# /inference 엔드포인트 정의
@app.post("/inference", response_model=ServiceOutput)
async def inference(request: ServiceInput):
    text = request.messages[0].text
    print(f"Message: f{text}") # For debugging
    response = run_inference(text)
    return {"text": response}



@app.post("/evaluation", response_model=EvaluationOutput)
async def evaluation(request: EvaluationInput):
    query = request.query

    contexts = retrieve_contexts(query)

    answer = generate_answer(query, contexts)

    return {"context": contexts, "answer": answer}


    # # 요청에서 메시지 가져오기
    # messages = request.messages
    # print(f"Message: f{messages[0].text}")
    
    # if not messages or not messages[0].text:
    #     raise HTTPException(status_code=400, detail="No message text provided")
    
    # input_text = messages[0].text
    # print(input_text)
    # model_response = run_inference(input_text)  # QA 함수 호출
    
    # return {"text": model_response}


# FastAPI 서버 실행 명령어
# uvicorn main:app --port 8000 --reload
# cloudflared tunnel --url http://localhost:8000