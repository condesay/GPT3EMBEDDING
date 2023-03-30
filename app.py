import streamlit as st
from transformers import pipeline

# Créer une fonction pour la classification de texte en utilisant le modèle RoBERTa
def classify_text(model, text, labels):
    classifier = pipeline('text-classification', model=model)
    result = classifier(text, labels=labels)[0]
    label = result['label']
    score = result['score']
    return label, score

# Créer une fonction main pour gérer l'exécution du programme
def main():
    st.title("Text Classification")
    text = st.text_area("Enter text to classify")
    label1 = st.selectbox("Select label 1", ["Positive", "Negative"])
    label2 = st.selectbox("Select label 2", ["Positive", "Negative"])
    if st.button("Classify"):
        model_name = "textattack/roberta-base-SST-2"
        labels = [label1, label2]
        label, score = classify_text(model_name, text, labels)
        st.write(f"The text is classified as {label} with a confidence score of {score}.")

if __name__ == "__main__":
    main()
