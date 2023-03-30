import streamlit as st
from transformers import pipeline

# Créer une fonction pour la classification de texte en utilisant le modèle RoBERTa
def classify_text(model, text):
    classifier = pipeline('sentiment-analysis', model=model)
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    return label, score

# Créer une fonction main pour gérer l'exécution du programme
def main():
    st.title("Text Classification")
    text = st.text_area("Enter text to classify")
    model_name = "textattack/roberta-base-SST-2"
    if st.button("Classify"):
        label, score = classify_text(model_name, text)
        st.write(f"The text is classified as {label} with a confidence score of {score}.")

if __name__ == "__main__":
    main()
