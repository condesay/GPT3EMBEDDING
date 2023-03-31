import openai
import re
import requests
import sys
from num2words import num2words
import os
import pandas as pd
import numpy as np
from openai.api import Api
from openai.models import Davinci, Curie, Babbage, Ada
from openai.encoder import SentenceEncoder
from transformers import GPT2TokenizerFast
import streamlit as st

API_KEY = os.getenv("OPENAI_API_KEY") 
openai.api_key = API_KEY

# Fonction pour extraire le score de similarité de la réponse générée par GPT-3
def extract_score(response):
    match = re.search(r"\d+\.\d+", response)
    if match:
        return match.group()
    else:
        return "Pas de similarité trouvé."

# Fonction pour récupérer la similarité entre deux textes en utilisant l'API OpenAI GPT-3
def get_similarity(text1, text2, model_engine):
    response = openai.Completion.create(
        engine=model_engine,
        prompt=f"Compare the similarity between these two texts:\n\nText 1: {text1}\n\nText 2: {text2}\n\nSimilarity:",
        max_tokens=64,
        n=1,
        stop=None,
        temperature=0.5,
        api_version="2021-10-01-preview"
    )
    similarity = response.choices[0].text.strip()
    similarity_score = extract_score(similarity)
    return similarity_score

# Fonction principale pour gérer l'exécution du programme
def main():
    st.title("Similarité entre textes")
    model_engine = st.sidebar.selectbox("Choisissez le modèle GPT-3", ["davinci", "curie", "babbage", "ada"])
    text1 = st.text_area("Texte 1")
    text2 = st.text_area("Texte 2")
    if st.button("Comparer"):
        similarity_score = get_similarity(text1, text2, model_engine)
        st.write(f"Le score de similarité entre les deux textes est {similarity_score}.")

if __name__ == "__main__":
    main()
