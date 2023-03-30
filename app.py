import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# fonction pour effectuer la classification de texte
def classify_text(text, labels, model_name):
    # charger le tokenizer et le modèle pré-entraîné
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # encodage du texte en entrée
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    # classification du texte
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)

    # retourner le label prédit
    return labels[predictions]

# fonction main pour exécuter le programme
def main():
    st.title("Text Classification")
    text = st.text_area("Enter some text to classify")
    label1 = st.text_input("Enter label 1:")
    label2 = st.text_input("Enter label 2:")
    if st.button("Classify"):
        # classification du texte saisi
        labels = [label1, label2]
        model_name = "distilbert-base-uncased-finetuned-sst-2-english" # modèle pré-entraîné pour la classification binaire
        label = classify_text(text, labels, model_name)
        st.write(f"The text is classified as '{label}'.")

if __name__ == "__main__":
    main()
