


# 20190827112156

import requests

API_URL = "https://api-inference.huggingface.co/models/anoukflinkert/time2warc-roberta"
headers = {"Authorization": "Bearer hf_YOUR_ACTUAL_TOKEN_HERE"} 

print("Pinging Hugging Face API...")
try:
    # The timeout prevents it from hanging forever
    response = requests.post(
        API_URL, 
        headers=headers, 
        json={"inputs": "Welcome to my retro 1990s web page!"}, 
        timeout=30 
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Connection Failed: {e}")