import os

import pg8000
import sqlalchemy
from dotenv import load_dotenv
from google.cloud.sql.connector import Connector, IPTypes
from sqlalchemy import text

load_dotenv()


def connect_with_connector() -> sqlalchemy.engine.base.Engine:
    """
    Initializes a connection pool for a Cloud SQL instance of Postgres.

    Uses the Cloud SQL Python Connector package.
    """
    # Note: Saving credentials in environment variables is convenient, but not
    # secure - consider a more secure solution such as
    # Cloud Secret Manager (https://cloud.google.com/secret-manager) to help
    # keep secrets safe.

    instance_connection_name = os.environ[
        "INSTANCE_CONNECTION_NAME"
    ]  # e.g. 'project:region:instance'
    db_user = os.environ["DB_USER"]  # e.g. 'my-db-user'
    db_pass = os.environ["DB_PASS"]  # e.g. 'my-db-password'
    db_name = os.environ["DB_NAME"]  # e.g. 'my-database'

    ip_type = IPTypes.PUBLIC

    # initialize Cloud SQL Python Connector object
    connector = Connector()

    # 해당 줄에서 에러가 발생된다면, GCP에서 사용자 추가할 것
    def getconn() -> pg8000.dbapi.Connection:
        conn: pg8000.dbapi.Connection = connector.connect(
            instance_connection_name,
            "pg8000",
            user=db_user,
            password=db_pass,
            db=db_name,
            ip_type=ip_type,
        )
        return conn

    # The Cloud SQL Python Connector can be used with SQLAlchemy
    # using the 'creator' argument to 'create_engine'
    pool = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
    )

    print("Success Connect!")
    return pool.connect()


def get_embedding(engine):
    query = text(
        f"""
WITH similar_docs AS (
    SELECT paragraph_id
    FROM embedding
)
SELECT
    p.paragraph_id,
    r.company_name,
    r.stockfirm_name,
    r.report_id,
    r.report_date,
    p.paragraph_text AS paragraph_text,
    ARRAY_AGG(DISTINCT t.tabular_text) FILTER (WHERE t.tabular_text IS NOT NULL) AS tabular_texts,
    ARRAY_AGG(DISTINCT i.image_text) FILTER (WHERE i.image_text IS NOT NULL) AS image_texts
FROM similar_docs sd
JOIN paragraph p 
    ON sd.paragraph_id = p.paragraph_id
JOIN report r 
    ON p.report_id = r.report_id
LEFT JOIN tabular t 
    ON p.paragraph_id = t.paragraph_id
LEFT JOIN image i 
    ON p.paragraph_id = i.paragraph_id
GROUP BY 
    r.company_name,
    r.stockfirm_name,
    r.report_id,
    r.report_date,
    p.paragraph_id,
    p.paragraph_text
"""
    )
    results = engine.conn.execute(query)
    return [result for result in results.mappings()]
