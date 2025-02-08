from typing import List

from FlagEmbedding import BGEM3FlagModel
from langchain_core.documents import Document
from sqlalchemy import text

from src.utils.load_engine import SqlEngine


class EmbeddingModel:
    _instance = None  # singleton 패턴 사용

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "model"):
            self.model = None

    def load_model(self):
        if self.model is None:
            print("임베딩 모델 로드 중...")
            self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
            print("모델 로드 완료")

    def get_embedding(self, sentence: str, max_length: int = 8192) -> list[float]:
        if self.model is None:
            raise RuntimeError(
                "모델이 아직 로드되지 않았습니다. 먼저 load_model()을 실행하세요."
            )

        if isinstance(sentence, str):
            return self.model.encode(sentence, max_length=max_length)["dense_vecs"]
        else:
            raise ValueError("입력은 문자열이어야 합니다.")

    def run_dense_retrieval(
        self, engine: SqlEngine, query_vector: List[float], paragraph_ids: List[str]
    ):
        vector_array = f"[{','.join(str(x) for x in query_vector)}]"
        joined_ids = (
            ", ".join(f"'{str(u)}'" for u in paragraph_ids)
            if len(paragraph_ids) > 0
            else ""
        )
        query = text(
            f"""
WITH similar_docs AS (
    SELECT paragraph_id, clean_text_embedding_vector
    FROM embedding
)
SELECT
    p.paragraph_id,
    r.company_name,
    r.stockfirm_name,
    r.report_id,
    r.report_date,
    p.paragraph_text AS paragraph_text
FROM similar_docs sd
JOIN paragraph p 
    ON sd.paragraph_id = p.paragraph_id
JOIN report r 
    ON p.report_id = r.report_id
{"WHERE p.paragraph_id in (" if joined_ids != "" else ""}{joined_ids}{")" if joined_ids != "" else ""}
GROUP BY
    r.company_name,
    r.stockfirm_name,
    r.report_id,
    r.report_date,
    p.paragraph_id,
    p.paragraph_text,
    sd.clean_text_embedding_vector
ORDER BY sd.clean_text_embedding_vector <-> '{vector_array}'::vector
{"limit 10" if joined_ids == "" else ""}
"""
        )
        results = engine.conn.execute(query)

        row_to_documents = []
        for row in results.mappings():
            metadata = {
                k: v
                for k, v in zip(row._key_to_index, row._data)
                if k != "paragraph_text"
            }
            row_to_documents.append(
                Document(page_content=row["paragraph_text"], metadata=metadata)
            )
        return row_to_documents


# if __name__ == "__main__":
#     print("임베딩 테스트")
#     print("종료는 ctrl+c")
#     embedding_model = EmbeddingModel()
#     embedding_model.load_model()

#     while(1):
#         sentence = input()
#         length = len(sentence) * 10
#         print(type(length), length)
#         print(f"임베딩 길이: {len(embedding_model.get_embedding(sentence, max_length=length))}\n")
