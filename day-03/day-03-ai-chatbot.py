import requests

url = "http://localhost:11434/api/generate"
model = "llama3"
conversation = []

def send_post_request(message):
    response = requests.post(
        url,
        json={
            'model': model,
            'prompt': message,
            'stream': False
        }
    )
    return response.json().get('response')

def my_chat_bot():
    while True:
        user_input = input('You: ')
        if user_input == 'quit' or user_input == 'bye':
            print(f'AI: Goodbye! \nHistory: {conversation}')
            break
        conversation.append(f'User: {user_input}')
        full_prompt = "\n".join(conversation)

        answer = send_post_request(full_prompt)
        ai_dialog = f'AI: {answer}'
        conversation.append(ai_dialog)
        print(ai_dialog)

my_chat_bot()