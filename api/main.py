from flask import Flask, request, jsonify
from clarifai.client.model import Model
from clarifai.client.input import Inputs
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

app = Flask(__name__)

# Obtenir la clé API depuis les variables d'environnement
CLARIFAI_PAT = os.getenv('CLARIFAI_PAT')
MODEL_URL = 'https://clarifai.com/openai/chat-completion/models/openai-gpt-4-vision'

@app.route('/api/formation', methods=['POST'])
def formation():
    # Récupérer les données de `form-data`
    image_url = request.form.get('image_url')
    prompt = request.form.get('prompt')

    # Exemple de logique pour varier l'URL de l'image
    if image_url == 'special_case':
        image_url = 'https://example.com/special_image.jpg'

    if not image_url or not prompt:
        return jsonify({'error': 'Image URL and prompt are required'}), 400

    inference_params = dict(temperature=0.2, max_tokens=100)
    multi_inputs = Inputs.get_multimodal_input(input_id="", image_url=image_url, raw_text=prompt)

    model_prediction = Model(url=MODEL_URL, pat=CLARIFAI_PAT).predict(
        inputs=[multi_inputs],
        inference_params=inference_params,
    )

    response_text = model_prediction.outputs[0].data.text.raw

    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
