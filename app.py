from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

app = Flask(__name__)

# Charger le modèle
model = load_model("model/CNN_fruits.h5")

# Les classes
class_names = ['apple', 'banana', 'orange']

# Dossier pour enregistrer les images
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}


# Fonction pour vérifier l'extension du fichier
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Fonction de prédiction
def predict(image):
    # Convertir l'image en RGB
    image = image.convert('RGB')
    img = image.resize((32, 32))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    return class_names[class_index], confidence


# Route principale
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Vérifier si un fichier a été téléchargé
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Vérifier si le fichier est autorisé
        if file and allowed_file(file.filename):
            # Enregistrer le fichier
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Charger l'image
            try:
                image = Image.open(filename)
                class_name, confidence = predict(image)
                return jsonify({
                    "class_name": class_name,
                    "confidence": f"{confidence * 100:.2f}%",
                    "image_url": f"/{filename}"  # URL pour afficher l'image
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        else:
            return jsonify({"error": "Invalid file type"}), 400

    return render_template("index.html")  # Interface simple pour tester


# Lancer l'application Flask
if __name__ == '__main__':
    # Créer le dossier uploads s'il n'existe pas
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True)
