from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import openai

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize Blip model
processor = BlipProcessor.from_pretrained("noamrot/FuseCap")
model = BlipForConditionalGeneration.from_pretrained("noamrot/FuseCap").to(device)

# Load API key for OpenAI
API_KEY = open("API_KEY", "r").read()
client = openai.OpenAI(api_key=API_KEY, organization="org-C3HZ989amK6IhNBeinAIDnML")


# Define function to generate questions based on image description
def generate_questions(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    text = "a picture of "
    inputs = processor(raw_image, text, return_tensors="pt").to(device)
    out = model.generate(**inputs, num_beams=3)
    description = processor.decode(out[0], skip_special_tokens=True)

    # Request questions based on the description from GPT-3.5
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Give me three questions based on this description: {description}"
            }
        ]
    )

    # Extract and return the questions from the response
    generated_questions = response.choices[0].message.content
    return generated_questions


# Example usage
image_path = 'images/sleepy_cat.jfif'
questions = generate_questions(image_path)
print("Generated questions:")
print(questions)
