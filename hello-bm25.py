from src.bm25retriever_with_scores import BM25RetrieverWithScores
from src.utils.gcp import connect_with_connector, get_embedding
from src.utils.utils import clean_text, create_documents

if __name__ == "__main__":
    try:
        # Test connection
        conn = connect_with_connector()
        texts = get_embedding(conn)
        # texts = open('sample.txt', 'r').read().split('"""\n"""')
        
        documents = create_documents(texts)
        retriever = BM25RetrieverWithScores.from_documents(documents, k=10)
        print(f"Created Okapi BM25 retriever")
        
        query = "CJ제일제당의 2025E 매출액은?"
        query = clean_text(query)
        
        docs = retriever.invoke(query)
        for doc, score in docs:
            print(f"Content written by {doc.metadata['stockfirm_name']} about {doc.metadata['company_name']} at {str(doc.metadata['report_date'])}")
            print(f"Score: {score}")
            print(doc.page_content + "\n" * 3)
        print("End of elastic")

    except Exception as e:
        print(f"Connection error: {e}")
