from typing import List, Tuple

from langchain_community.document_transformers import LongContextReorder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sqlalchemy import UUID, Engine

from src.utils.gcp import connect_with_connector, get_embedding
from src.utils.utils import clean_korean_text, create_documents


class BM25RetrieverWithScores(BM25Retriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Tuple[Document, float]]:
        processed_query = self.preprocess_func(query)
        scores = self.vectorizer.get_scores(processed_query)
        docs_with_scores = list(zip(self.docs, scores))
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        return docs_with_scores[: self.k]


class CustomizedOkapiBM25:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "retriever"):
            self.retriever = None

    def load_retriever(self, conn: Engine, k: int = 10):
        if self.retriever is None:
            print("Loading BM25 Retriever")
            texts = get_embedding(conn)
            self.documents = create_documents(texts)
            self.retriever = BM25RetrieverWithScores.from_documents(
                documents=self.documents, k=k
            )

    def get_pids_from_sparse(self, query: str):
        query = clean_korean_text(query)
        docs = self.retriever.invoke(query)
        self.docs = []
        self.scores = []
        for doc, score in docs:
            self.docs.append(doc)
            self.scores.append(score)
        paragraph_ids = [doc.metadata["paragraph_id"] for doc in self.docs]
        return paragraph_ids

    def get_contexts_by_ids(self, ids: List[UUID], limit):
        uuid_pos_map = {uid: i for i, uid in enumerate(ids)}

        def sort_key(doc: Document) -> int:
            # If doc's UUID is not in the map, return a large number so it ends up at the end
            return uuid_pos_map.get(doc.metadata["paragraph_id"], float("inf"))

        # Sort the documents based on their position in `ids`
        sorted_docs = sorted(self.docs, key=sort_key)[:limit]
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(sorted_docs)
        contexts = []
        for doc in reordered_docs:
            contexts.append(
                {
                    "company_name": doc.metadata["company_name"],
                    "report_date": doc.metadata["report_date"],
                    "paragraph_text": doc.metadata["raw_text"],
                }
            )
        return contexts
