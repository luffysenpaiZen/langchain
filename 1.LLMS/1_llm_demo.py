from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm=OpenAI(model='gpt3-turbo',temperature=0,max_tokens=10)
result=llm.invoke("what is the capital of andhra pradesh")
print(result)