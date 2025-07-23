from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-pro',temperature=0.3)
result=model.invoke("what is the capital of Andhra pradesh ?")

print(result.content)