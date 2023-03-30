import openai
import streamlit as st

# Créer une fonction pour récupérer la clé API OpenAI GPT-3 saisie par l'utilisateur
def get_api_key():
    api_key = st.text_input("Entrez votre clé OpenAI API:")
    return api_key

# Créer une fonction pour récupérer la similarité entre deux textes en utilisant l'API OpenAI GPT-3
def get_similarity(text1, text2, model_engine):
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

# Créer une fonction main pour gérer l'exécution du programme
def main():
    st.title("Recherche de Similarité entre deux textes")
    api_key = get_api_key()
    if api_key:
        openai.api_key = api_key
        model_engine = "text-similarity-davinci-001"
        text1 = st.text_area("Texte 1")
        text2 = st.text_area("Texte 2")
        if st.button("Comparez"):
            similarity = get_similarity(text1, text2, model_engine)
            st.write(f"La similarité entre les deux textes est {similarity}.")

if __name__ == "__main__":
    main()