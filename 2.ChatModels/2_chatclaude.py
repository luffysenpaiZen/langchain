from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model=ChatAnthropic(
	model_name='claude-opus-4-20250514',
	temperature=0.8,
	timeout=30,  # example timeout in seconds
	stop=None    # example stop sequence, set to None if not needed
)
result=model.invoke("what is the capital of india?")

print(result.content)