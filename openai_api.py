import openai

API_KEY = open("API_KEY", "r").read()

client = openai.OpenAI(api_key=API_KEY, organization="org-C3HZ989amK6IhNBeinAIDnML")

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "Give me three questions based on this description: a picture of a gray cat "
                       "with green eyes and a brown nose sits on a white bed with a white pillow, "
                       "against a white wall its pointy ears are visible on either side of its"
        }
    ]
)

print(response)
