from model import model

response = model.invoke(
    "My name is Murtaja."
)

print("First response:")
print(response.content)


response = model.invoke(
    "What is my name?"
)

print("\nSecond response:")
print(response.content)
