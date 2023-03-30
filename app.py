import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Créer une fonction main pour gérer l'exécution du programme
def main():
    st.title("Text Classification App")

    # Demander à l'utilisateur d'entrer les deux labels
    label_1 = st.text_input("Enter label 1:")
    label_2 = st.text_input("Enter label 2:")

    # Demander à l'utilisateur d'entrer le texte à classer
    text = st.text_area("Enter text:")

    # Charger le modèle BERT pré-entraîné pour la classification de séquences
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Transformer le texte en entrée en un encodage numérique qui peut être compris par le modèle BERT
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    # Obtenir les scores de classification à partir du modèle BERT
    outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).tolist()[0]

    # Déterminer le label avec le score le plus élevé et l'afficher à l'utilisateur
    if scores[0] > scores[1]:
        st.write(f"The text is classified as {label_1} with a confidence score of {scores[0]:.2f}.")
    else:
        st.write(f"The text is classified as {label_2} with a confidence score of {scores[1]:.2f}.")

if __name__ == "__main__":
    main()
