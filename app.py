import openai
import streamlit as st

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
            similarity_score = float(similarity)
            st.write(f"The similarity score between the two texts is {similarity_score:.2f}.")

if __name__ == "__main__":
    main()
