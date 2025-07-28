import schedule
import streamlit as st
import time as tm
from schedule import every,repeat
from datetime import time,timedelta,datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# llm
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint



load_dotenv()
llm=HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-R1",
    task='text-generation',
    temperature=0
)

model=ChatHuggingFace(llm=llm)

model2=ChatGoogleGenerativeAI(model='gemini-2.5-pro',temperature=0)


@repeat(every().day.at("11:00"))
def stock_updater():
    print('working')
    result=model.invoke('what are recent financial and stocks updates all over the world?')
    print(result.content)

count=0
while True:
    schedule.run_pending()
    
    tm.sleep(1)
    count+=1
    
    

    