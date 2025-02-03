from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from text_embedding import EmbeddingModel  # 임베딩 모델 (bge-m3) 불러오기
from load_engine import SqlEngine
from typing import List
from sqlalchemy import text
load_dotenv()


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's request only based on the given context."),
    ("user", "Context: {context}\nQuestion: {question}\nAnswer:")
])
model = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()
chain = prompt | model | output_parser


embedding_model = EmbeddingModel()
embedding_model.load_model()

engine = SqlEngine()


def RAG(query: str) -> str:
    
    # TODO: 임베딩 함수 구현 (bge m3 사용하기)
    embedding = embedding_model.get_embedding(query)

    # TODO: db에서 추출하는 함수 구현 (함수 input, output 설계)
    # INPUT: {임베딩, k} OUTPUT: 문자열 리스트
    #context = GET_CONTEXT_FROM_DB(embedding, k)

    # DEMO (하드코딩)
    # context: List[str] = ['''2024년 12월 04일 I 기업분석_기업분석(Report)
    # CJ제일제당 (097950)
    # 선택과 집중 긍정적 + 가격 매력
    # BUY (유지)
    # 목표주가(12M) 480,000원 
    # 현재주가(12.03) 271,500원''',
    # '''Key Data
    # KOSPI 지수 (pt) 2,500.10
    # 52주 최고/최저(원) 398,000/240,500
    # 시가총액(십억원) 4,087.2
    # 시가총액비중(%) 0.20
    # 발행주식수(천주) 15,054.2
    # 60일 평균 거래량(천주) 45.9
    # 60일 평균 거래대금(십억원) 12.9
    # 외국인지분율(%) 23.78
    # 주요주주 지분율(%)
    # CJ 외 8 인 45.51
    # 국민연금공단 11.86
    # Consensus Data
    # <table>
    # 2024 2025
    # 매출액(십억원) 29,494.8 30,537.6
    # </table>''',]
    context = search_similar_content(embedding, 3)
    
    return context


def run_inference(query: str) -> str:
    query = query.strip()
    try:
        if not query:
            raise ValueError("query is empty or only contains whitespace.")
        

        # TODO: RAG 함수 구현
        context = RAG(query)
        context = format_retrieved_docs(context)

        #response = chain.invoke({"question": query, "context": context})
        response = context  #TEST CODE
        
        if not response or not response.strip():
            raise ValueError("The model returned an empty or invalid response.")
        
        return response
    except Exception as e:
        print(f"Error during inference: {e}")
        return "An error occurred during inference. Please try again later."


def format_retrieved_docs(docs):
    # RAG된 docs Formatting 진행
    formatted_docs = []
    for doc in docs:
        text = f"""
Company: {doc['company_name']}
Date: {doc['report_date']}
Content: {doc['paragraph_text']}
"""
        if doc['is_tabular']:
            text += f"Table: {doc['tabular_text']}\n"
        if doc['is_image']:
            text += f"Image Content: {doc['image_text']}\n"
        formatted_docs.append(text)
    
    return "\n---\n".join(formatted_docs)

def search_similar_content(query_vector, limit: int = 5):
    vector_array = f"[{','.join(str(x) for x in query_vector)}]"
    query = text(f"""
    WITH similar_docs AS (
        SELECT paragraph_id 
        FROM embedding
        ORDER BY text_embedding_vector <-> '{vector_array}'::vector
        LIMIT {limit}
    )
    SELECT 
        r.company_name,
        r.stockfirm_name, 
        r.report_date,
        p.paragraph_text,
        p.is_tabular,
        p.is_image,
        t.tabular_text,
        i.image_text
    FROM similar_docs sd
    JOIN paragraph p ON sd.paragraph_id = p.paragraph_id
    JOIN report r ON p.report_id = r.report_id
    LEFT JOIN tabular t ON p.paragraph_id = t.paragraph_id AND p.is_tabular = True
    LEFT JOIN image i ON p.paragraph_id = i.paragraph_id AND p.is_image = True
    """)
    results = engine.connect().execute(query)
    return [dict(row._mapping) for row in results]



def retrieve_contexts(query: str) -> list[str]:
    query_vector = embedding_model.get_embedding(query)
    limit = 3

    vector_array = f"[{','.join(str(x) for x in query_vector)}]"
    query = text(f"""
    WITH similar_docs AS (
        SELECT paragraph_id 
        FROM embedding
        ORDER BY text_embedding_vector <-> '{vector_array}'::vector
        LIMIT {limit}
    )
    SELECT 
        r.company_name,
        r.stockfirm_name, 
        r.report_date,
        p.paragraph_text,
        p.is_tabular,
        p.is_image,
        t.tabular_text,
        i.image_text
    FROM similar_docs sd
    JOIN paragraph p ON sd.paragraph_id = p.paragraph_id
    JOIN report r ON p.report_id = r.report_id
    LEFT JOIN tabular t ON p.paragraph_id = t.paragraph_id AND p.is_tabular = True
    LEFT JOIN image i ON p.paragraph_id = i.paragraph_id AND p.is_image = True
    """)
    results = engine.connect().execute(query)
    docs = [dict(row._mapping) for row in results]

    formatted_docs = []
    for doc in docs:
        _text = f"""
        Company: {doc['company_name']}
        Date: {doc['report_date']}
        Content: {doc['paragraph_text']}
        """
        if doc['is_tabular']:
            _text += f"Table: {doc['tabular_text']}\n"
        if doc['is_image']:
            _text += f"Image Content: {doc['image_text']}\n"
        formatted_docs.append(_text)
    return formatted_docs

def generate_answer(query: str, contexts: list[str]) -> str:
    context = "\n---\n".join(contexts)
    answer = chain.invoke({"question": query, "context": context})
    return answer