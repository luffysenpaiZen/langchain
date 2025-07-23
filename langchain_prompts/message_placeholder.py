from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# model
model=ChatGoogleGenerativeAI(model='gemini-2.5-pro',temperature=0.3)

# create a template
chat_template=ChatPromptTemplate([
    ('system','You are a helpful customer service and information agent '),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history=[]

# load chat_history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())
    
print(chat_history)

# create a prompt
prompt=chat_template.invoke({'chat_history':chat_history,'query':"where is my refund"})

# result

result=model.invoke(prompt)

print(result.content)
