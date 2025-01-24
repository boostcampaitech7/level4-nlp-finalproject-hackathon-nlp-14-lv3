from dotenv import load_dotenv
from create_table import connect_with_connector, create_table, check_table_exists
from data_insert import process_excel_and_insert_data

load_dotenv()

if __name__ == "__main__":
    # Connect to the database
    engine = connect_with_connector()

    # Create tables
    create_table(engine)

    # Verify the tables exist
    check_table_exists(engine, "report")
    check_table_exists(engine, "paragraph")
    check_table_exists(engine, "tabular")
    check_table_exists(engine, "image")
    check_table_exists(engine, "embedding")

    # Define base directory
    BASE_DIR = "../data/랩큐"

    # Process and insert data
    process_excel_and_insert_data(engine, BASE_DIR)