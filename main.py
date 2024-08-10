from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import requests
from groq import Groq

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

app = Flask(__name__)

# Obtenir les clés API depuis les variables d'environnement
CLARIFAI_PAT = os.getenv('CLARIFAI_PAT')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

MODEL_URL = 'https://clarifai.com/openai/chat-completion/models/openai-gpt-4-vision'

@app.route('/api/formation', methods=['POST'])
def formation():
    try:
        image_url = request.form.get('image_url')
        prompt = request.form.get('prompt')

        if not image_url or not prompt:
            return jsonify({'error': 'Image URL and prompt are required'}), 400

        # Test de l'URL de l'image
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to access the image URL'}), 400

        # Préparation des entrées pour le modèle
        inference_params = dict(temperature=0.2, max_tokens=100)
        multi_inputs = Inputs.get_multimodal_input(input_id="", image_url=image_url, raw_text=prompt)

        # Prédiction du modèle
        model_prediction = Model(url=MODEL_URL, pat=CLARIFAI_PAT).predict(
            inputs=[multi_inputs],
            inference_params=inference_params,
        )

        response_text = model_prediction.outputs[0].data.text.raw
        return jsonify({'response': response_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/query', methods=['GET'])
def query_prompt():
    prompt = request.args.get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    try:
        # Utilisation de Groq Mistral AI avec clé API
        client = Groq(api_key=GROQ_API_KEY)  # Incluez la clé API ici
        completion = client.chat.completions.create(
            model="gemma2-9b-it",  # Spécifiez le modèle que vous utilisez
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        # Vérifier si la complétion est un tuple et en extraire le contenu
        if isinstance(completion, tuple):
            completion = completion[0]

        # Extraction du texte de réponse
        response_text = ""
        for chunk in completion['choices']:
            response_text += chunk['message']['content']

        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
