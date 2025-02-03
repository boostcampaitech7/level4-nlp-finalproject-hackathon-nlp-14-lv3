import os

from google.cloud.sql.connector import Connector, IPTypes
import pg8000

import sqlalchemy
from sqlalchemy import Table, Column, MetaData, ForeignKey, VARCHAR, DATE, TEXT, ARRAY, BOOLEAN, UUID, inspect
from pgvector.sqlalchemy import Vector

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

    #해당 줄에서 에러가 발생된다면, GCP에서 사용자 추가할 것
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
    return pool


def create_table(engine):
    """Create tables in the database with updated data types and relationships."""

    metadata = MetaData()

    # Report table
    report = Table(
        "report",
        metadata,
        Column("report_id", UUID, primary_key=True),  
        Column("company_name", VARCHAR(30), nullable=False),  
        Column("stockfirm_name", VARCHAR(30), nullable=False),  
        Column("report_date", DATE, nullable=False),  
    )

    # Paragraph table
    # paragraph_text에 null이 들어가면, 해당 표 혹은 img는 특정 문단이 없는 것으로 간주
    paragraph = Table(
        "paragraph",
        metadata,
        Column("paragraph_id", UUID, primary_key=True),  
        Column("report_id", UUID, ForeignKey("report.report_id"), nullable=False),  # Foreign key to report
        Column("paragraph_text", TEXT, nullable=True),  
        Column("is_tabular", BOOLEAN, nullable=False, default=False),  # Indicates if a tabular exists
        Column("is_image", BOOLEAN, nullable=False, default=False),  # Indicates if an image exists
    )

    # Tabular table
    tabular = Table(
        "tabular",
        metadata,
        Column("tabular_id", UUID, primary_key=True),  
        Column("paragraph_id", UUID, ForeignKey("paragraph.paragraph_id"), nullable=False),  # Foreign key to paragraph
        Column("tabular_text", TEXT, nullable=False),  
    )

    # Image table
    image = Table(
        "image",
        metadata,
        Column("image_id", UUID, primary_key=True), 
        Column("paragraph_id", UUID, ForeignKey("paragraph.paragraph_id"), nullable=False),  # Foreign key to paragraph 
        Column("image_text", TEXT, nullable=False), 
    )

    # Embedding table
    embedding = Table(
        "embedding",
        metadata,
        Column("embedding_id", UUID, primary_key=True),  
        Column("paragraph_id", UUID, ForeignKey("paragraph.paragraph_id"), nullable=False),  # Foreign key to paragraph
        Column("text_embedding_vector", Vector(1024), nullable=False),  # Embedding vector
        Column("tabular_id", ARRAY(UUID), nullable=True),  # Array of tabular IDs
        Column("image_id", ARRAY(UUID), nullable=True),  # Array of image IDs
    )

    # Create tables in the database
    metadata.create_all(engine)
    print("Tables created successfully!")


def check_table_exists(engine, table_name):
    """Check if a table exists in the database."""
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    if table_name in tables:
        print(f"Table '{table_name}' exists in the database.")
    else:
        print(f"Table '{table_name}' does not exist in the database.")
