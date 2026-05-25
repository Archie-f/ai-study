def user_chat():
    messages = []
    while True:
        user_input = input('Your message: ')

        if user_input == 'quit' or user_input == 'bye':
            break
        messages.append(user_input)

        print(f'History : {messages}')

user_chat()