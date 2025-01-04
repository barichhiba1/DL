from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

# Initialiser l'application Flask
app = Flask(__name__)

# Charger le modèle
model = load_model("model/CNN_fruits.h5")

# Classes cibles
class_names = ['apple', 'banana', 'orange']


# Fonction de prédiction
def predict(image):
    image = image.convert('RGB')  # Convertir en RGB
    img = image.resize((32, 32))  # Redimensionner l'image
    img_array = img_to_array(img) / 255.0  # Normaliser
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    return class_names[class_index], confidence


# Route pour la page principale
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Vérifier si un fichier a été uploadé
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            # Charger l'image
            image = Image.open(file)
            # Prédire la classe
            class_name, confidence = predict(image)
            return render_template(
                "index.html",
                class_name=class_name,
                confidence=f"{confidence * 100:.2f}%",
                uploaded_image=file.filename
            )

    return render_template("index.html")


# Lancer l'application
if __name__ == "__main__":
    app.run(debug=True)
