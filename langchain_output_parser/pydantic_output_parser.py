from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    model='Qwen/Qwen3-Coder-480B-A35B-Instruct',
    task='text-generation',
    temperature=1.5
    
)

model=ChatHuggingFace(
    llm=llm
)

class format(BaseModel):
    name:str=Field(description='Name of the person')
    age:int=Field(gt=18,description='age of the person ')
    city:str=Field(description='city of the perso living now')
    
parser=PydanticOutputParser(pydantic_object=format)

template=PromptTemplate(
    template='generate name,age,city of a fictional character from this {place} \n{format_instructions}',
    input_variables=['place'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

# prompt=template.invoke({'place':'india'})

# result=model.invoke(prompt)

# final_result=parser.parse(result.content) #type: ignore

# print(final_result)

chain=template|model|parser

result=chain.invoke({'place':'india'})

print(result)
