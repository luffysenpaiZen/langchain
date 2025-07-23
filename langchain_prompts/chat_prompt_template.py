from langchain_core.prompts import ChatPromptTemplate

chat_template=ChatPromptTemplate([
    ('system','you are professional and helpful {domain} expert '),
    ('human','explain me  about this {topic}')
    
])

prompt=chat_template.invoke({'domain':'Cricket','topic':'googly'})

print(prompt)