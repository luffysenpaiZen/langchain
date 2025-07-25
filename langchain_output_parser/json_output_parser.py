from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts  import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import time
load_dotenv()

llm=HuggingFaceEndpoint(
    model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    task='text-generation',
    temperature=0.5
)

model=ChatHuggingFace(llm=llm)

# parser
parser=JsonOutputParser()

template=PromptTemplate(
    template='give me 5 facts about this {topic} \n {format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
    
)

prompt=template.invoke({'topic':'black hole'})

result=model.invoke(prompt)

final_result1=parser.parse(result.content) # type: ignore


# chaining

chain=template | model | parser

final_result2=chain.invoke({'topic':'black hole'})



print(final_result1)

time.sleep(1)

print(final_result2)


