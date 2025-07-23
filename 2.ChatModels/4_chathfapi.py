from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()

llm=HuggingFaceEndpoint(
    task="text-generation",
    model="deepseek-ai/DeepSeek-R1",
    temperature= 0.6,
    top_p=0.95,
    max_new_tokens=10
)

model=ChatHuggingFace(llm=llm)

result=model.invoke('what is the capital of india?')

print(result.content)