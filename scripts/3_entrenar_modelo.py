import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm

# Configuración
BATCH_SIZE = 32  # Puedes probar 32 si va bien
EPOCHS = 1       # Cambia esto a 2 o 3 más tarde si todo va bien
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Usando dispositivo: {device}")

# Cargar dataset tokenizado
print("📦 Cargando dataset tokenizado...")
data = torch.load("data/dataset_tokenizado.pt")

train_inputs = data["train_encodings"]["input_ids"]
train_masks = data["train_encodings"]["attention_mask"]
train_labels = data["train_labels"]

test_inputs = data["test_encodings"]["input_ids"]
test_masks = data["test_encodings"]["attention_mask"]
test_labels = data["test_labels"]

# Crear DataLoader
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Modelo BERT
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# Entrenamiento
print("🚀 Entrenando modelo...")
model.train()
for epoch in range(EPOCHS):
    print(f"🔁 Época {epoch+1}/{EPOCHS}")
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        b_input_ids, b_input_mask, b_labels = [x.to(device) for x in batch]

        outputs = model(
            input_ids=b_input_ids,
            attention_mask=b_input_mask,
            labels=b_labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

    # Guardar el modelo al final de cada época
    model_path = f"models/modelo_epoca{epoch+1}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"💾 Modelo guardado: {model_path}")

print("✅ Entrenamiento completado.")