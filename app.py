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
    text1 = st.text_area("Enter text 1")
    text2 = st.text_area("Enter text 2")
    label1 = st.text_input("Enter label 1")
    label2 = st.text_input("Enter label 2")
    if st.button("Classify"):
        model_name = "textattack/roberta-base-SST-2"
        labels = [label1, label2]
        label1_score = classify_text(model_name, text1, labels)[1]
        label2_score = classify_text(model_name, text2, labels)[1]
        if label1_score > label2_score:
            st.write(f"Text 1 is classified as {label1} with a confidence score of {label1_score}.")
            st.write(f"Text 2 is classified as {label2} with a confidence score of {label2_score}.")
        else:
            st.write(f"Text 1 is classified as {label2} with a confidence score of {label2_score}.")
            st.write(f"Text 2 is classified as {label1} with a confidence score of {label1_score}.")

if __name__ == "__main__":
    main()
