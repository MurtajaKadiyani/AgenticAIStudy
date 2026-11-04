from model import model

messages = [
    {   
        "role": "user",
        "content": "My name is Murtaja."
    }
]

first_response = model.invoke(messages)

messages.append(
    {   
       "role": "assistant",
       "content": first_response.content # type: ignore
    }
)

messages.append(
    {   
        "role": "user",
        "content": "What is my name?"
    }
)

second_response = model.invoke(messages)
print(second_response.content)