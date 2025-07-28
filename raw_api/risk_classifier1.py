import os
import mysql.connector
from dotenv import load_dotenv
from typing import Literal
from decimal import Decimal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
load_dotenv()


DB_CONFIG = {
    'user': 'root',       
    'password':os.getenv('DB_password'),  #password
    'host': 'localhost',
    'database': 'financial_risk_db'
}


class RiskClassification(BaseModel):
    """Data model for a loan application risk classification."""
    applicant_name: str = Field(description="The full name of the applicant")
    risk_category: Literal['Low Risk', 'Moderate Risk', 'High Risk'] = Field(description="The classified risk category")
    justification: str = Field(description="A brief justification for the classification decision, based on the data.")


def fetch_loan_applications():
    """Connects to MySQL and fetches all loan applications."""
    applications = []
    try:
        print("Connecting to the MySQL database...")
        cnx = mysql.connector.connect(**DB_CONFIG)
        
        cursor = cnx.cursor(dictionary=True) 
        
        query = "SELECT * FROM Loan_applications"
        cursor.execute(query)
        
        applications = cursor.fetchall()
        print(f"Successfully fetched {len(applications)} applications.")
        
    except mysql.connector.Error as err:
        print(f"Error fetching data from MySQL: {err}")
    finally:
        if 'cnx' in locals() and cnx.is_connected():
            cursor.close()
            cnx.close()
            print("MySQL connection is closed.")
    return applications

def main():
    """Main function to run the classification process."""
    # Fetch the data
    applicants = fetch_loan_applications()
    if not applicants:
        print("No applicants to process. Exiting.")
        return

    # llm
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

    #parser for our structured output
    parser = PydanticOutputParser(pydantic_object=RiskClassification)

    #Prompt Template
    prompt = PromptTemplate(
        template="""
        You are an expert financial analyst specializing in risk assessment for loan applications.
        Your task is to analyze the following loan application data and classify it into one of three risk categories: Low Risk, Moderate Risk, or High Risk.

        Provide a brief justification for your decision based on factors like credit score, debt-to-income ratio, and employment status.

        {format_instructions}

        Here is the application data:
        Applicant Name: {applicant_name}
        Credit Score: {credit_score}
        Annual Income: ${annual_income:,.2f}
        Debt-to-Income Ratio: {debt_to_income_ratio:.2%}
        Employment Status: {employment_status}
        Years Employed: {years_employed}
        Loan Amount Requested: ${loan_amount:,.2f}
        Loan Purpose: {loan_purpose}
        Number of Existing Loans: {existing_loans_count}
        Has Recent Bankruptcies: {recent_bankruptcies}
        """,
        input_variables=[
            "applicant_name", "credit_score", "annual_income", "debt_to_income_ratio",
            "employment_status", "years_employed", "loan_amount", "loan_purpose",
            "existing_loans_count", "recent_bankruptcies"
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    #LangChain chain
    chain = prompt | llm | parser

    print("\n--- Starting Loan Application Classification ---\n")
    # Loop through each applicant
    for applicant_data in applicants:
        try:
            
            # Pre-process the applicant data
            input_for_prompt = {
                "applicant_name": applicant_data["applicant_name"], # type: ignore
                "credit_score": applicant_data["credit_score"], # type: ignore
                "annual_income": float(applicant_data["annual_income"]), # type: ignore
                "debt_to_income_ratio": float(applicant_data["debt_to_income_ratio"]), # type: ignore
                "employment_status": applicant_data["employment_status"], # type: ignore
                "years_employed": applicant_data["years_employed"], # type: ignore
                "loan_amount": float(applicant_data["loan_amount"]), # type: ignore
                "loan_purpose": applicant_data["loan_purpose"], # type: ignore
                "existing_loans_count": applicant_data["existing_loans_count"], # type: ignore
                "recent_bankruptcies": "Yes" if applicant_data["recent_bankruptcies"] else "No" # type: ignore
            }

            
            result = chain.invoke(input_for_prompt)
            
            # Print the structured result
            print(f"Applicant: {result.applicant_name}")
            print(f"Risk Category: {result.risk_category}")
            print(f"Justification: {result.justification}")
            print("-" * 30)

        except Exception as e:
            print(f"Could not process applicant {applicant_data.get('applicant_name', 'N/A')}. Error: {e}") # type: ignore
            print("-" * 30)


if __name__ == "__main__":
    main()
