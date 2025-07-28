from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    model='deepseek-ai/DeepSeek-R1',
    task='text-generation',
    temperature=0.8
)

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

template1=PromptTemplate(
    template='write a joke on this {topic}',
    input_variables=['topic']
)

template2=PromptTemplate(
    template='write a short essay on this {topic}',
    input_variables=['topic']
)

seq1=RunnableSequence(template1,model,parser)
seq2=RunnableSequence(template2,model,parser)

run_parallel=RunnableParallel({
    'joke':seq1,
    'essay':seq2
})

result=run_parallel.invoke({'topic':'AI'})

print(result['joke'])

