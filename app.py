import openai
import streamlit as st
import re

# Fonction pour extraire le score de similarité de la réponse générée par GPT-3
def extract_score(response):
    match = re.search(r"\d+\.\d+", response)
    if match:
        return match.group()
    else:
        return "No similarity score found."

# Fonction pour récupérer la clé API OpenAI GPT-3 saisie par l'utilisateur
def get_api_key():
    api_key = st.text_input("Enter your OpenAI API key:")
    return api_key

# Fonction pour récupérer la similarité entre deux textes en utilisant l'API OpenAI GPT-3
def get_similarity(text1, text2, model_engine):
    prompt = f"Compare the similarity between these two texts:\n\nText 1: {text1}\n\nText 2: {text2}\n\nSimilarity:"
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=64,
        n=1,
        stop=None,
        temperature=0.5,
    )
    similarity = response.choices[0].text.strip()
    similarity_score = extract_score(similarity)
    return similarity_score

# Fonction principale pour gérer l'exécution du programme
def main():
    st.title("Text Similarity Checker")
    api_key = get_api_key()
    if api_key:
        openai.api_key = api_key
        model_engine = "text-similarity-davinci-001"
        text1 = st.text_area("Text 1")
        text2 = st.text_area("Text 2")
        if st.button("Compare"):
            similarity_score = get_similarity(text1, text2, model_engine)
            st.write(f"The similarity score between the two texts is {similarity_score}.")

if __name__ == "__main__":
    main()
