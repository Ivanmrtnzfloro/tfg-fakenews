import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification
from sklearn.metrics import classification_report
from tqdm import tqdm

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Usando dispositivo: {device}")

# Cargar dataset
print("📦 Cargando dataset tokenizado...")
data = torch.load("data/dataset_tokenizado.pt")

test_inputs = data["test_encodings"]["input_ids"]
test_masks = data["test_encodings"]["attention_mask"]
test_labels = data["test_labels"]

test_dataset = TensorDataset(test_inputs, test_masks, test_labels)
test_loader = DataLoader(test_dataset, batch_size=16)

# Cargar modelo
print("📥 Cargando modelo entrenado...")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.load_state_dict(torch.load("models/modelo_epoca1.pt", map_location=device))
model.to(device)
model.eval()

# Evaluación
print("Evaluando...")
preds = []
true_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluando"):
        b_input_ids, b_input_mask, b_labels = [x.to(device) for x in batch]

        outputs = model(
            input_ids=b_input_ids,
            attention_mask=b_input_mask
        )

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

        preds.extend(predictions.cpu().numpy())
        true_labels.extend(b_labels.cpu().numpy())

# Resultados
print("📊 Resultados de la evaluación:")
print(classification_report(true_labels, preds, digits=4))
