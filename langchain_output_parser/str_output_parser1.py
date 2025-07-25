from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


llm=HuggingFaceEndpoint(
    model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    task='text-generation'
)

model=ChatHuggingFace(llm=llm)
# template1
template1=PromptTemplate(
    template='Write a detailed report on this {topic}',
    input_variables=['topic']
    
)

# template2

template2=PromptTemplate(
    template='Write a 5 line summary on the following {text}',
    input_variables=['text']
    
)


# strOutput parser

parser=StrOutputParser()

chain = template1 | model | parser | template2 | model | parser


result=chain.invoke({'topic':'blackhole'})


print(result)