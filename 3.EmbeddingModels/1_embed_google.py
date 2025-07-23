from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding=GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-exp-03-07')

docs=[
    "hi my name is issac",
    "hello your name is gemini"
    "Amaravathi is the capital of Andhra pradesh"
]


result1=embedding.embed_query("Amaravathi is the capital of Andhra pradesh")
result2=embedding.embed_documents(docs)
print(str(result1))
print(str(result2))