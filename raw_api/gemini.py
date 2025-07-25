import requests
from dotenv import load_dotenv
import os
load_dotenv()

API_KEY = os.getenv('GOOGLE_API_KEY')
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
headers = {"Content-Type": "application/json"}

def get_completion(prompt,API_URL,system_prompt="You are a helpful assistant."):
    data = {
        "contents": [
            
            {"role": "user", "parts": [{"text":prompt}]}
            
        ],
        "systemInstruction": {
            "role": "system",
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 500,
            "topP": 0.9,
            "topK": 40,
        },
        
    }

    try:
        response = requests.post(API_URL, json=data, headers=headers)
        response.raise_for_status()
        reply = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return reply
        
    except Exception as e:
        return f"An error occurred: {e}"
system_query = "You are a helpful assistant living in Bengaluru. Your name is 'Nandi'. It is currently a pleasant Friday afternoon."
user_query = "can you say me your name"
answer = get_completion(user_query,API_URL,system_query)
print(f"User: {user_query}")
print(f"Gemini: {answer}")