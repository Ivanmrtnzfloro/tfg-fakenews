
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Cargar el modelo
model_path = "models/modelo_epoca1.pt"
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Cargar tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Función para predecir
def predecir_fake_news(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    return "❌ Noticia FALSA" if pred == 1 else "✅ Noticia VERDADERA"

# Entrada por terminal
print("🔎 Detección de Fake News")
while True:
    texto = input("\n📝 Introduce un texto (o 'salir' para terminar):\n> ")
    if texto.lower() == "salir":
        break
    resultado = predecir_fake_news(texto)
    print(f" Resultado del modelo: {resultado}")
