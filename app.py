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
    st.title(" Similarité entre textes")
    text1 = st.text_area("Texte 1")
    text2 = st.text_area("Texte 2")
    if st.button("Compare"):
        similarity_score = get_similarity(text1, text2)
        st.write(f"Le Score de Similarité entre les deux textes est: {similarity_score}.")

if __name__ == "__main__":
    main()
