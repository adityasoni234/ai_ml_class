import requests
import json

url = "https://api.sampleapis.com/coffee/hot"

response = requests.get(url)
data = json.loads(response.text)

for i in data:
    print(i["title"])