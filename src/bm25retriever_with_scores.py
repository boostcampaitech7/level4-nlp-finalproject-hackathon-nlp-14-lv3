from typing import List, Tuple
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

class BM25RetrieverWithScores(BM25Retriever):
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Tuple[Document, float]]:
        # Preprocess the query (tokenize, etc.)
        processed_query = self.preprocess_func(query)
        # Here we assume that your BM25 vectorizer has a method to get scores;
        # for example, you might call get_scores(processed_query, self.docs)
        # (Note: this is not implemented by default in BM25Retriever.)
        scores = self.vectorizer.get_scores(processed_query)  # hypothetical method
        # Pair each document with its score
        docs_with_scores = list(zip(self.docs, scores))
        # Sort by score in descending order and take the top k results
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        return docs_with_scores[: self.k]