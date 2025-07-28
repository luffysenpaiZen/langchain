from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    model='deepseek-ai/DeepSeek-R1',
    task='text-generation',
    temperature=0.5
    
)

model=ChatHuggingFace(llm=llm)

template=PromptTemplate(
    template='write a short essay on this {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()

sequence=RunnableSequence(template,model,parser)

result=sequence.invoke({'topic':'AI'})

print(result)