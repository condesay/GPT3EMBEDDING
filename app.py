import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Créer une fonction pour charger le modèle et le tokenizer
@st.cache(allow_output_mutation=True)
def load_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Créer une fonction main pour gérer l'exécution du programme
def main():
    st.title("Document Classification")
    
    # Sélectionner le modèle à utiliser
    model_name = st.selectbox("Select a model", ["bert-base-uncased", "textattack/roberta-base-SST-2"])
    
    # Charger le modèle et le tokenizer
    model, tokenizer = load_model(model_name)
    
    # Entrer le texte à classifier
    text = st.text_area("Enter some text")
    
    # Sélectionner les classes à classifier
    class1 = st.text_input("Enter class 1")
    class2 = st.text_input("Enter class 2")
    
    # Classer le texte lorsque l'utilisateur clique sur le bouton "Classify"
    if st.button("Classify"):
        # Encoder le texte en utilisant le tokenizer
        encoded_text = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')
        
        # Faire une prédiction en utilisant le modèle
        with torch.no_grad():
            outputs = model(encoded_text['input_ids'], token_type_ids=None, attention_mask=encoded_text['attention_mask'])
            prediction = torch.argmax(outputs[0]).item()
        
        # Afficher le résultat
        if prediction == 0:
            st.write(class1)
        else:
            st.write(class2)

if __name__ == "__main__":
    main()
