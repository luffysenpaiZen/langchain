from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda,RunnableSequence,RunnableParallel,RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    model='deepseek-ai/DeepSeek-R1',
    task='text-generation',
    temperature=0.5
)

model=ChatHuggingFace(llm=llm)

parser= StrOutputParser()

template1=PromptTemplate(
    template='write one joke on this {topic}',
    input_variables=['topic']
)

seq1=RunnableSequence(template1,model,parser)

parallel=RunnableParallel(
    {
        'joke':RunnablePassthrough(),
        'length':RunnableLambda(lambda x:len(x.split()))
    }
)

seq2=RunnableSequence(seq1,parallel)

result=seq2.invoke({'topic':'AI'})

print(result)
