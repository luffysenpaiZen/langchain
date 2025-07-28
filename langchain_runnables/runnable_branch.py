from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda,RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableBranch
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    model='deepseek-ai/DeepSeek-R1',
    task='text-generation',
    temperature=0.5
)

model=ChatHuggingFace(llm=llm)

parser= StrOutputParser()

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)
report_gen_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, prompt2 | model | parser), #type: ignore
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukraine'}))