
import streamlit as st
import pandas as pd
import openai

# Définir les informations d'identification pour l'API OpenAI GPT-3
openai.api_key = "YOUR_API_KEY"

# Définir le modèle de similarité
model_engine = "text-similarity-davinci-001"

# Créer une fonction qui utilise l'API OpenAI GPT-3 pour calculer la similarité entre deux textes
def get_similarity(text1, text2):
    response = openai.Completion.create(
        engine=model_engine,
        prompt=f"Compare the similarity between these two texts:\n\nText 1: {text1}\n\nText 2: {text2}\n\nSimilarity:",
        max_tokens=64,
        n=1,
        stop=None,
        temperature=0.5,
    )
    similarity = response.choices[0].text.strip()
    return similarity

# Définir l'interface utilisateur Streamlit
st.title("Text Similarity Checker")
st.write("Enter two texts to compare their similarity.")

text1 = st.text_area("Text 1")
text2 = st.text_area("Text 2")

if st.button("Compare"):
    similarity = get_similarity(text1, text2)
    st.write(f"The similarity between the two texts is {similarity}.")
