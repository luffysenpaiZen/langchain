from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    model='Qwen/Qwen3-Coder-480B-A35B-Instruct',
    task='text-generation'  
)

model=ChatHuggingFace(llm=llm)

# template 1
template1=PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
    
)

# template 2

template2=PromptTemplate(
    template='Write a 5 line summary on the following text. \n {text}',
    input_variables=['text']
)


prompt1=template1.invoke({'topic':'blackhole'})

result1=model.invoke(prompt1)

prompt2=template2.invoke({'text':result1.content})

result2=model.invoke(prompt2)

print(result2.content)
