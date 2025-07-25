from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.runnables import RunnableLambda,RunnableBranch,RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    model='mistralai/Mistral-7B-Instruct-v0.2',
    task='text-generation'
)

model=ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    sentiment: Literal['positive','negative']=Field(description='give the sentiment of the text')

parser1=StrOutputParser()
parser2=PydanticOutputParser(pydantic_object=Feedback)


prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)



prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)
sentiment_analyzer = RunnablePassthrough.assign(
    classification_result=prompt1 | model | parser2
)
branch_chain = RunnableBranch(
    (lambda x:x['classification_result'].sentiment == 'positive', prompt2 | model | parser1),  # type: ignore
    (lambda x:x['classification_result'].sentiment == 'negative', prompt3 | model | parser1),  # type: ignore
    RunnableLambda(lambda x: "could not find sentiment")
)

chain=sentiment_analyzer | branch_chain

result=chain.invoke({'feedback':'This is a baahubali  phone'})

print(result)

chain.get_graph().print_ascii()

