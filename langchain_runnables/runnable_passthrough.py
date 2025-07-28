from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableSequence,RunnablePassthrough
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
    template='explain this joke {joke}',
    input_variables=['joke']
)

seq1=RunnableSequence(template1,model,parser)

parallel=RunnableParallel(
    {
        'joke':RunnablePassthrough(),
        'explanation':RunnableSequence(template2,model,parser)
    }
)

seq2=RunnableSequence(seq1,parallel)

result=seq2.invoke({'topic':'AI'})

print(result)