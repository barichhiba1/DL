import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image


#st.write("write testest")

#charger le modele
model=load_model("model/CNN_fruits.h5")

#les classes
class_names=['apple','banana','orange']

#fonction de prediction
#tout ce qu'on a fait dans le modele doit etre fait ici
def predict(image):
     # Convertir l'image en RGB (au cas où elle serait RGBA ou en mode autre que RGB)
    image = image.convert('RGB')
    img=image.resize((32,32))
    img_array=img_to_array(img)/255.0
    img_array=np.expand_dims(img_array,axis=0)
    predictions=model.predict(img_array)
    class_index=np.argmax(predictions)
    confidence=np.max(predictions)
    return class_names[class_index],confidence


#interface streamlit
st.title("CNN modele pour classification des fruits")
st.write("Ce modele va predire si le fruit est une banane , pomme ou orange :")

#charger l'image
uploaded_file=st.file_uploader("Telecharger l'image", type=['png','jpg','jpeg'])

if uploaded_file:
    image=Image.open(uploaded_file)
    st.image(image,caption="Image telechargé", use_container_width=True) #utilise la taille du conteneur
    #Prediction
    #spinner : affiche un message en attendant la fin de l'analyse
    with st.spinner("Analyse en cours"):
        class_name,confidence=predict(image)
        st.success(f"Résultat : {class_name}({confidence*100:.2f}%)")