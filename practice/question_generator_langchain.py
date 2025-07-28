from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from dotenv import load_dotenv


load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-pro')

chat_template=ChatPromptTemplate([
    ('system','you are helpful multi choice questions generator based on short question and answers'),
        ('human','can you please generate ten multi choice questions  having four options on the {questions}.They should have one or two correct options and the remaining two or three options should be related to question but wrong') # type: ignore
    ] 
)

template=PromptTemplate(
    template='{questions}',
    input_variables=['questions']
)
questions="""1. You notice a job is failing with an “OutOfMemoryError”. How would you debug
this in Spark?
Start by checking the Spark UI for the stage where the failure occurs. Look at the
memory usage and task duration. You might need to increase executor-memory and
check if caching is being overused. Evaluate if any wide transformations are causing
large shuffles and memory pressure. Consider increasing the number of partitions or
using broadcast joins if applicable. Always ensure unnecessary data isn’t being
persisted.
2. How do you handle skewed data in a Spark job?
Skewed data often causes performance bottlenecks because some tasks take
significantly longer. To resolve this, you can apply salting techniques to spread hot keys
across multiple reducers. Another approach is to use broadcast joins when one dataset
is significantly smaller. Custom partitioning strategies or pre-aggregating skewed keys
can also help reduce skew impact.
3. How can you persist intermediate results and which storage levels are optimal?
Use .cache() when data fits in memory or .persist(StorageLevel.MEMORY_AND_DISK) if
not. Caching is useful when the same DataFrame or RDD is reused across multiple
actions. Persisting intermediate results helps avoid recomputation and can significantly
reduce execution time, especially in iterative algorithms. Choose storage levels based
on your memory constraints."""

chat_prompt=chat_template.invoke({'questions':questions})
prompt=template.invoke({'questions':questions})
result=model.invoke(chat_prompt)

print(result.content)