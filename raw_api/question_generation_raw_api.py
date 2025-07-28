import requests
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api=os.getenv('GOOGLE_API_KEY')
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={gemini_api}"
headers = {"Content-Type": "application/json"}

def generateQuestions(prompt,API_URL,system_prompt):
    data={
        "contents":[
            {'role':'user',
             'parts':[{'text':prompt}]
            }
        ],
        "system_instruction": {
            "role": "system",
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.9,
            "topK": 40,
        },
    }
    try:
        response = requests.post(API_URL, json=data, headers=headers)
        
        # # ADD THESE THREE LINES FOR DEBUGGING
        # print(f"Status Code: {response.status_code}")
        # print("Full API Response:")
        # print(response.text) # Using .text is safer in case the response isn't valid JSON

        response.raise_for_status()
        reply = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return reply
        
    except Exception as e:
        return f"An error occurred: {e}"
    
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

system_query = "you are helpful multi choice questions generator based on short question and answers"
user_query = f'can you please generate ten multi choice questions  having four options on the {questions}.They should have one or two correct options and the remaining two or three options should be related to question but wrong'
answer = generateQuestions(user_query,API_URL,system_query)
print(f"User: {user_query}")
print(f"Gemini: {answer}")