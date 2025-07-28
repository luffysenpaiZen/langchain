from langchain_community.document_loaders import DirectoryLoader
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

loader=DirectoryLoader(
    path=r'langchain_loaders\books',
    glob='*.pdf'
)

docs=loader.load()

template=PromptTemplate(
    template='write a short summary on this {page}',
    input_variables=['page']
)

parser=StrOutputParser()

chain=template | model | parser

result=chain.invoke({'page':docs[0].page_content})

print(result)