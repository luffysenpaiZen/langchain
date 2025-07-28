from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    model='deepseek-ai/DeepSeek-R1',
    task='text-generation',
    temperature=0.3
)

model=ChatHuggingFace(llm=llm)

loader=TextLoader(file_path='langchain_loaders/cricket.txt',encoding='utf-8')

parser=StrOutputParser()

docs=loader.load()

prompt=PromptTemplate(
    template='what is the topic this poem is about \n {poem}',
    input_variables=['poem']
)

chain=prompt | model | parser

result=chain.invoke({'poem':docs[0].page_content})

print(result)