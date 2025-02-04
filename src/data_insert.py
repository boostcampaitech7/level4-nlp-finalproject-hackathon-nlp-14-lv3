import os
import pandas as pd
import uuid
from sqlalchemy import insert
from sqlalchemy.orm import sessionmaker
from sqlalchemy import MetaData
from text_embedding import EmbeddingModel 

def process_excel_and_insert_data(engine, base_dir):
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

    # Initialize the embedding model ###
    embedding_model = EmbeddingModel()
    embedding_model.load_model()

    # Iterate through each company directory
    for company_name in os.listdir(base_dir):
        company_path = os.path.join(base_dir, company_name)
        if not os.path.isdir(company_path):
            continue

        # Process each Excel file in the company directory
        for excel_file in os.listdir(company_path):
            if excel_file.endswith(".xlsx"):
                # Generate UUID for report_id
                report_id = str(uuid.uuid4())

                # Extract stockfirm_name and date from filename
                _, stockfirm_name, date_part = os.path.splitext(excel_file)[0].split("_")
                report_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}"
                report_date = pd.to_datetime(report_date).date()

                # Insert into report table
                report_insert = {
                    "report_id": report_id,
                    "company_name": company_name,
                    "stockfirm_name": stockfirm_name,
                    "report_date": report_date,
                }
                session.execute(insert(report_table), report_insert)

                # Load Excel file
                excel_path = os.path.join(company_path, excel_file)
                df = pd.read_excel(excel_path)

                # Process each row in the Excel file
                for idx, row in df.iterrows():
                    text = row["text"]
                    embedding_text = row["embedding"]

                    # Generate UUID for paragraph_id
                    paragraph_id = str(uuid.uuid4())

                    # Insert into paragraph table and commit
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

                    # Insert into embedding table
                    embedding_insert = {
                        "embedding_id": str(uuid.uuid4()),
                        "paragraph_id": paragraph_id,
                        "text_embedding_vector": embedding_vector,
                    }
                    session.execute(insert(embedding_table), embedding_insert)

    # Commit transaction
    session.commit()
    print("Data inserted successfully!")