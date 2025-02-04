from typing import List

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sqlalchemy import text

from src.bm25retriever_with_scores import CustomizedOkapiBM25
from src.text_embedding import EmbeddingModel  # 임베딩 모델 (bge-m3) 불러오기
from src.utils.load_engine import SqlEngine

load_dotenv()


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Please respond to the user's request only based on the given context.",
        ),
        ("user", "Context: {context}\nQuestion: {question}\nAnswer:"),
    ]
)
model = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()
chain = prompt | model | output_parser


embedding_model = EmbeddingModel()
embedding_model.load_model()

engine = SqlEngine()

sparse_retriever = CustomizedOkapiBM25()
sparse_retriever.load_retriever(engine, k=5)


def RAG(query: str, spr_limit: int = 10, dpr_limit: int = 3) -> str:
    """
    query_embedding: 문자열 query의 임베딩 결과

    paragraph_ids: BM25 기반 문자열 query와 유사도가 높은 문서들의 paragraph_id 리스트
    - BM25 문서 갯수는 sparse_retriever.load_retriever(engine, k=5)에서 k값 조절

    ordered_ids: paragraph_id를 재정렬한 리스트
    - BM25 유사도가 높은 문서들 중에서 query_embedding 벡터와 유사도가 가장 높은 문서 순서대로
    - dpr_limit 갯수만큼만 문서를 가져옴

    contexts: 원본 텍스트들의 리스트
    - paragraph_id 리스트에서 재정렬를 실행하고 이에 대응하는 원본 텍스트를 가져옴



    의문점)
    지금은 임베딩 벡터를 원본 텍스트에서 숫자만 제거한거로 만들고 있는데
    임베딩 하기전에 한국어 전처리로 단어제거(불용어 등), 특수문자 제거, stem 단어 변환 등을 하는게 더 좋을 것 같아요 -> 비교를 해보죠 저의들의 평가 데이터셋으로 제거하는 과정은 얼마걸리지 않으니깐

    """
    query_embedding = embedding_model.get_embedding(query)
    paragraph_ids = sparse_retriever.get_pids_from_sparse(query)
    ordered_ids = run_dense_retriever(query_embedding, paragraph_ids)
    contexts = sparse_retriever.get_contexts_by_ids(ordered_ids, dpr_limit)
    return contexts


def run_inference(query: str) -> str:
    query = query.strip()
    try:
        if not query:
            raise ValueError("query is empty or only contains whitespace.")

        contexts = RAG(query)
        context = format_retrieved_docs(contexts)
        response = chain.invoke({"question": query, "context": context})

        if not response or not response.strip():
            raise ValueError("The model returned an empty or invalid response.")

        return response
    except Exception as e:
        print(f"Error during inference: {e}")
        return "An error occurred during inference. Please try again later."


def run_evaluation(query: str) -> str:
    query = query.strip()
    try:
        if not query:
            raise ValueError("query is empty or only contains whitespace.")

        context = RAG(query)
        contexts, joined_context = format_retrieved_docs(
            context, return_single_str=False
        )
        response = chain.invoke({"question": query, "context": joined_context})

        if not response or not response.strip():
            raise ValueError("The model returned an empty or invalid response.")

        return {"context": contexts, "answer": response}
    except Exception as e:
        print(f"Error during inference: {e}")
        return "An error occurred during inference. Please try again later."


def format_retrieved_docs(docs: List[str], return_single_str=True):
    # RAG된 docs Formatting 진행
    formatted_docs = []
    for doc in docs:
        text = f"""Company: {doc['company_name']} Date: {doc['report_date']} Content: {doc['paragraph_text']}"""
        formatted_docs.append(text)
    joined_formatted_docs = "\n---\n".join(formatted_docs)
    if return_single_str:
        return joined_formatted_docs
    else:
        return formatted_docs, joined_formatted_docs


def run_dense_retriever(query_vector: List[float], paragraph_ids: List[str]):
    vector_array = f"[{','.join(str(x) for x in query_vector)}]"
    joined_ids = ", ".join(f"'{str(u)}'" for u in paragraph_ids)
    query = text(
        f"""
SELECT paragraph_id 
FROM embedding
WHERE paragraph_id in ({joined_ids})
ORDER BY text_embedding_vector <-> '{vector_array}'::vector
"""
    )
    results = engine.conn.execute(query)
    return [row for row in results]


if __name__ == "__main__":
    while True:
        query = input("질의를 입력해주세요: ")
        result = run_inference(query)
        print(result)
