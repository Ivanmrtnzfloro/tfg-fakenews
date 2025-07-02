from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Inicializar la app de Flask, especificando la carpeta de templates
app = Flask(__name__, template_folder='../templates')

# Cargar modelo y tokenizer
model_path = "models/modelo_epoca1.pt"
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Página principal
@app.route('/')
def home():
    return render_template("index.html")

# Ruta para hacer la predicción
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    # Tokenizar el texto
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # Hacer la predicción
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    # Interpretar resultado
    resultado = "✅ Noticia REAL" if predicted_class == 0 else "❌ Noticia FALSA"

    return render_template("index.html", resultado=resultado, texto=text)

# Ejecutar la app
if __name__ == '__main__':
    app.run(debug=True)

