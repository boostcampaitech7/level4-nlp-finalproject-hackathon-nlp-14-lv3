from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional
from inference import run_inference  # QA (inference) 함수 불러오기
from text_embedding import EmbeddingModel  # 임베딩 모델 (bge-m3) 불러오기
from load_engine import connect_with_connector
from sqlalchemy.engine.base import Engine
# FastAPI 앱 생성
app = FastAPI()

# 요청 데이터 모델 정의 
class MessageContent(BaseModel):
    role: Optional[str] = "ai"
    text: Optional[str] = None
    files: Optional[List[str]] = None
    html: Optional[str] = None

class InferenceRequest(BaseModel):
    messages: List[MessageContent]


# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 허용할 출처 추가 "http://localhost:8080", "http://localhost:5173"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding_model = EmbeddingModel()
    embedding_model.load_model()
    engine = connect_with_connector()
    app.state.embedding_model = embedding_model
    app.state.engine = engine
    yield


def get_engine(app: FastAPI = Depends()) -> Engine:
    return app.state.engine

def get_embedding_model(app: FastAPI = Depends()) -> EmbeddingModel:
    return app.state.embedding_model


# /inference 엔드포인트 정의
@app.post("/inference")
async def inference(request: InferenceRequest):
    # 요청에서 메시지 가져오기
    messages = request.messages
    
    if not messages or not messages[0].text:
        raise HTTPException(status_code=400, detail="No message text provided")
    
    input_text = messages[0].text
    print(input_text)
    model_response = run_inference(input_text)  # QA 함수 호출
    
    return {"text": model_response}

# FastAPI 서버 실행 명령어
# uvicorn main:app --port 8000 --reload
# cloudflared tunnel --url http://localhost:8000