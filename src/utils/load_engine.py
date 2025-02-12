import os

import pg8000
import sqlalchemy
from dotenv import load_dotenv
from google.cloud.sql.connector import Connector, IPTypes

load_dotenv()


class SqlEngine:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "conn"):
            instance_connection_name = os.environ["INSTANCE_CONNECTION_NAME"]
            db_user = os.environ["DB_USER"]
            db_pass = os.environ["DB_PASS"]
            db_name = os.environ["DB_NAME"]

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
            self.conn = pool.connect()
            print("Success Connect!")

    def connect(self):
        return self.conn
