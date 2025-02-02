from langchain_community.retrievers import BM25Retriever
from src.utils.gcp import connect_with_connector, get_embedding
from src.utils.utils import clean_text, create_documents

if __name__ == "__main__":
    try:
        # Test connection
        conn = connect_with_connector()
        print("Connected successfully!")
        texts = get_embedding(conn)
        # texts = open('sample.txt', 'r').read().split('"""\n"""')
        
        documents = create_documents(texts)
        retriever = BM25Retriever.from_documents(documents)
        print(f"Created Okapi BM25 retriever")
        
        query = "CJ제일제당의 2025E 매출액은?"
        query = clean_text(query)
        
        docs = retriever.invoke(query)
        for doc in docs:
            print(f"Content written by {doc.metadata['stockfirm_name']} about {doc.metadata['company_name']} at {str(doc.metadata['report_date'])}")
            print(doc.page_content + "\n" * 3)
        print("End of elastic")

    except Exception as e:
        print(f"Connection error: {e}")
