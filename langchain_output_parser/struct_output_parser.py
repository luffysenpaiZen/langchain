from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    model='Qwen/Qwen3-Coder-480B-A35B-Instruct',
    task='text-generation'
)

model=ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser=StructuredOutputParser.from_response_schemas(schema)

template=PromptTemplate(
    template='give 3 facts about this {topic} \n {format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
    
)

# prompt=template.invoke({'topic':'black hole'})

# result=model.invoke(prompt)

# final_result=parser.parse(result.content) #type: ignore

# print(final_result)


chain=template | model | parser

result=chain.invoke({'topic':'Newton'})

print(result)

