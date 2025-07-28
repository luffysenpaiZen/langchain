import mysql.connector
import os
from dotenv import load_dotenv
from mysql.connector import errorcode
load_dotenv()
DB_CONFIG = {
    'user': 'root',       # e.g., 'root'
    'password': os.getenv('DB_password'),
    'host': '127.0.0.1',           # Or your server's IP address
}

DB_NAME = 'financial_risk_db'
TABLE_NAME = 'Loan_applications'

loan_data = [
    ('Alice Johnson', 750, 85000.00, 0.25, 'Employed', 7, 150000.00, 'Home Renovation', 1, False),
    ('Bob Williams', 620, 50000.00, 0.45, 'Employed', 2, 25000.00, 'Debt Consolidation', 3, False),
    ('Charlie Brown', 810, 120000.00, 0.15, 'Self-Employed', 10, 500000.00, 'Business Expansion', 0, False),
    ('Diana Prince', 580, 40000.00, 0.55, 'Unemployed', 0, 10000.00, 'Emergency Funds', 2, True),
    ('Eve Adams', 700, 70000.00, 0.30, 'Employed', 5, 100000.00, 'Car Purchase', 1, False)
]

TABLES = {}
TABLES[TABLE_NAME] = (
    f"CREATE TABLE `{TABLE_NAME}` ("
    "  `application_id` int(11) NOT NULL AUTO_INCREMENT,"
    "  `applicant_name` varchar(255) NOT NULL,"
    "  `credit_score` int(11) NOT NULL,"
    "  `annual_income` decimal(15, 2) NOT NULL,"
    "  `debt_to_income_ratio` decimal(5, 2) NOT NULL,"
    "  `employment_status` varchar(50) NOT NULL,"
    "  `years_employed` int(11) NOT NULL,"
    "  `loan_amount` decimal(15, 2) NOT NULL,"
    "  `loan_purpose` text,"
    "  `existing_loans_count` int(11) NOT NULL,"
    "  `recent_bankruptcies` boolean NOT NULL,"
    "  PRIMARY KEY (`application_id`)"
    ") ENGINE=InnoDB"
)
def create_database(cursor):
    """Creates the database if it doesn't exist."""
    try:
        cursor.execute(f"CREATE DATABASE {DB_NAME} DEFAULT CHARACTER SET 'utf8'")
        print(f"Database '{DB_NAME}' created successfully.")
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_DB_CREATE_EXISTS:
            print(f"Database '{DB_NAME}' already exists.")
        else:
            print(f"Failed creating database: {err}")
            exit(1)

def main():
    """Main function to connect, create DB/Table, and insert data."""
    cnx = None
    cursor = None
    try:
        # Establish connection to the MySQL server
        cnx = mysql.connector.connect(**DB_CONFIG)
        cursor = cnx.cursor()
        print("Successfully connected to MySQL server.")

        # Create the database
        create_database(cursor)
        
        # Switch to our new database
        cursor.execute(f"USE {DB_NAME}")
        print(f"Using database '{DB_NAME}'.")

        # Create the table
        table_description = TABLES[TABLE_NAME]
        try:
            print(f"Creating table '{TABLE_NAME}': ", end='')
            cursor.execute(table_description)
            print("OK")
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                print("already exists.")
            else:
                print(err.msg)
        
        # Insert the data into the table
        print(f"Inserting data into '{TABLE_NAME}'...")
        
        insert_query = (
            f"INSERT INTO {TABLE_NAME} "
            "(applicant_name, credit_score, annual_income, debt_to_income_ratio, "
            "employment_status, years_employed, loan_amount, loan_purpose, "
            "existing_loans_count, recent_bankruptcies) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        
        # executemany is efficient for inserting multiple rows
        cursor.executemany(insert_query, loan_data)
        
        # Commit the changes to the database
        cnx.commit()
        
        print(f"{cursor.rowcount} records inserted successfully.")

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    finally:
        # Clean up the connection
        if cursor:
            cursor.close()
        if cnx and cnx.is_connected():
            cnx.close()
            print("MySQL connection is closed.")

if __name__ == "__main__":
    main()
