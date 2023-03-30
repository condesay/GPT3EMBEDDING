import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Créer une fonction pour récupérer la similarité entre deux textes en utilisant SentenceTransformer
def get_similarity(text1, text2):
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    embeddings = model.encode([text1, text2])
    similarity_score = util.cos_sim(embeddings[0], embeddings[1])
    return similarity_score

# Créer une fonction main pour gérer l'exécution du programme
def main():
    st.title("Text Similarity Checker")
    text1 = st.text_area("Text 1")
    text2 = st.text_area("Text 2")
    if st.button("Compare"):
        similarity_score = get_similarity(text1, text2)
        st.write(f"The similarity score between the two texts is {similarity_score}.")

if __name__ == "__main__":
    main()
