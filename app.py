from flask import Flask, render_template, request
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import openai
import os

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize Blip model
processor = BlipProcessor.from_pretrained("noamrot/FuseCap")
model = BlipForConditionalGeneration.from_pretrained("noamrot/FuseCap").to(device)

# Load API key for OpenAI
API_KEY = open("API_KEY", "r").read()
client = openai.OpenAI(api_key=API_KEY, organization="org-C3HZ989amK6IhNBeinAIDnML")

UPLOAD_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    if 'file' not in request.files:
        return render_template('index.html', no_file_part_error=True)

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', no_file_selected_error=True)

    # Save the uploaded image
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)

    # Process the uploaded image to generate questions
    raw_image = Image.open(img_path).convert('RGB')
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
    content = response.choices[0].message.content

    # Split the content string into individual questions
    generated_questions = content.split('\n')

    # Remove empty strings and leading/trailing whitespace
    generated_questions = [q.strip() for q in generated_questions if q.strip()]

    return render_template('index.html', generated_questions=generated_questions)


if __name__ == '__main__':
    app.run(debug=True)
