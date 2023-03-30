import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# fonction pour effectuer la classification de texte
def classify_text(text, labels, model_name):
    # charger le tokenizer et le modèle pré-entraîné
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # encode le texte en entrée et récupère les probabilités de chaque classe
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()[0]
    probs = torch.softmax(torch.tensor(logits), dim=0).tolist()

    # retourne le label avec la plus grande probabilité
    max_prob = max(probs)
    max_prob_idx = probs.index(max_prob)
    return labels[max_prob_idx]

# fonction main pour exécuter le programme
def main():
    st.title("Text Classification")
    text = st.text_area("Enter some text to classify")
    label1 = st.text_input("Enter label 1:")
    label2 = st.text_input("Enter label 2:")
    if st.button("Classify"):
        # classification du texte saisi
        labels = [label1, label2]
        model_name = "textattack/roberta-base-MRPC" # modèle pré-entraîné pour la classification
        label = classify_text(text, labels, model_name)
        st.write(f"The text is classified as '{label}'.")

if __name__ == "__main__":
    main()
