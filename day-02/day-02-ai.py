import requests

url = "http://localhost:11434/api/generate"

def generate_request_body(model, stream):
    prompt = input('>>>Enter prompt: ')
    return {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }

response = requests.post(url, json=generate_request_body(
    'llama3',
    False))

data = response.json()
print(data['response'])