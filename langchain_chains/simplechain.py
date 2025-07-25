from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    model='Qwen/Qwen3-Coder-480B-A35B-Instruct',
    task='text-generation',
    temperature=0.3
    
)

model=ChatHuggingFace(llm=llm)

template=PromptTemplate(
    template='give me 3 fun facts about this {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()

# simple chain

chain=template | model | parser

result=chain.invoke({'topic':'indian schooling'})

print(result)