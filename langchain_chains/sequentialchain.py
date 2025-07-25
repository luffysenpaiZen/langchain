from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-pro',temperature=0.3)

template1=PromptTemplate(
    template='give me a 10 lines description on this {topic}',
    input_variables=['topic']
)

template2=PromptTemplate(
    template='give me 5 facts on this {text}',
    input_variables=['text']
)

parser=StrOutputParser()

# sequential parser

chain=template1| model | parser | template2 | model | parser

result=chain.invoke({'topic':'black hole'})

print(result)

chain.get_graph().print_ascii()
