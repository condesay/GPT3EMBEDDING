import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import streamlit as st

# Fonction pour récupérer la clé API OpenAI GPT-3 saisie par l'utilisateur
def get_api_key():
    api_key = st.text_input("Enter your OpenAI API key:")
    return api_key

# Fonction pour récupérer la similarité entre deux textes en utilisant l'API OpenAI GPT-3
def get_similarity(text1, text2, model_engine):
    # Récupérer les vecteurs d'embedding pour chaque texte
    embedding1 = get_embedding(text1, model_engine)
    embedding2 = get_embedding(text2, model_engine)

    # Calculer le score de similarité en utilisant la fonction cosine_similarity
    similarity = cosine_similarity(embedding1, embedding2)

    return similarity

# Fonction principale pour gérer l'exécution du programme
def main():
    st.title("Text Similarity Checker")
    api_key = get_api_key()
    if api_key:
        openai.api_key = api_key
        model_engine = "text-babbage-001"
        text1 = st.text_area("Text 1")
        text2 = st.text_area("Text 2")
        if st.button("Compare"):
            similarity = get_similarity(text1, text2, model_engine)
            st.write(f"The similarity score between the two texts is {similarity}.")

if __name__ == "__main__":
    main()
