from dotenv import load_dotenv

from src.utils.create_table import (check_table_exists, connect_with_connector,
                                    create_table)
from src.utils.data_insert import process_excel_and_insert_data

load_dotenv()

if __name__ == "__main__":
    # Connect to the database
    engine = connect_with_connector()

    # Create tables
    create_table(engine)

    # Verify the tables exist
    check_table_exists(engine, "report")
    check_table_exists(engine, "paragraph")
    check_table_exists(engine, "embedding")

    # Define base directory, scripts 경로에서 실행할 경우 ../data/랩큐로 변경
    BASE_DIR = "../data/랩큐"

    # Process and insert data
    process_excel_and_insert_data(engine, BASE_DIR)
