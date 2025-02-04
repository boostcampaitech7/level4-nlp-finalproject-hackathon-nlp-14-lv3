import os
import io
import pandas as pd
import numpy as np
import uuid
import re

from sqlalchemy import insert
from sqlalchemy.orm import sessionmaker
from sqlalchemy import MetaData

from google.cloud import storage
from google.oauth2 import service_account

from dotenv import load_dotenv

from text_embedding import EmbeddingModel

load_dotenv()

def connect_to_gcs():
    credentials = service_account.Credentials.from_service_account_file(
        os.environ['JSON_FILE']
    )
    
    try:
        storage_client = storage.Client(credentials=credentials)
        print("Google Cloud Storage에 성공적으로 연결되었습니다.")
        return storage_client
    except Exception as e:
        print(f"Google Cloud Storage 연결 실패: {e}")
        return None


def process_excel_and_insert_data(engine, bucket_name):
    """
    Processes the Excel files and inserts data into the database.
    """
    Session = sessionmaker(bind=engine)
    session = Session()

    # Reflect the database schema
    metadata = MetaData()
    metadata.reflect(bind=engine)

    # Access the necessary tables
    report_table = metadata.tables["report"]
    paragraph_table = metadata.tables["paragraph"]
    embedding_table = metadata.tables["embedding"]

    # Initialize the embedding model
    embedding_model = EmbeddingModel()
    embedding_model.load_model()

    # Cloud Storage 연결
    storage_client = connect_to_gcs()
    bucket = storage_client.bucket(bucket_name)
    
    # GCS에서 data/ 경로 내 모든 파일 가져오기
    blobs = storage_client.list_blobs(bucket_name, prefix="data/")
    
    # 회사별 디렉토리 추출
    company_folders = set(blob.name.split("/")[1] for blob in blobs if len(blob.name.split("/")) > 2)

    for company_name in company_folders:
        excel_folder_path = f"data/{company_name}/excel/"
        
        # 해당 회사의 Excel 파일 목록 가져오기
        excel_files = [blob for blob in blobs 
                       if blob.name.startswith(excel_folder_path) and blob.name.endswith(".xlsx")]

        for excel_blob in excel_files:
            print(f"Processing file: {excel_blob.name}")
            
            # report_id 생성
            report_id = str(uuid.uuid4())

            # 파일명에서 증권사 이름과 날짜 추출
            filename = os.path.basename(excel_blob.name)
            stockfirm_name, date_part = os.path.splitext(filename)[0].split("_")
            report_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}"
            report_date = pd.to_datetime(report_date).date()

            # report 테이블에 데이터 삽입
            report_insert = {
                "report_id": report_id,
                "company_name": company_name,
                "stockfirm_name": stockfirm_name,
                "report_date": report_date,
            }
            session.execute(insert(report_table), report_insert)

            # Excel 파일 다운로드 및 로드
            excel_data = excel_blob.download_as_bytes()
            df = pd.read_excel(io.BytesIO(excel_data))

            # Excel 파일의 각 행 처리
            for idx, row in df.iterrows():
                text = row.get("text", None)
                embedding_text = row.get("embedding", None)

                # paragraph_id 생성
                paragraph_id = str(uuid.uuid4())

                # paragraph 테이블에 데이터 삽입
                paragraph_insert = {
                    "paragraph_id": paragraph_id,
                    "report_id": report_id,
                    "paragraph_text": text if text else "None",
                }
                session.execute(insert(paragraph_table), paragraph_insert)

                # Generate embedding vector for embedding_text
                embedding_vector = None
                if embedding_text and isinstance(embedding_text, str):
                    embedding_vector = embedding_model.get_embedding(embedding_text)

                # embedding 테이블에 데이터 삽입 (현재 기본 벡터 사용)
                embedding_insert = {
                    "embedding_id": str(uuid.uuid4()),
                    "paragraph_id": paragraph_id,
                    "text_embedding_vector": embedding_vector,
                }
                session.execute(insert(embedding_table), embedding_insert)

    # 트랜잭션 커밋
    session.commit()
    print("Data inserted successfully!")