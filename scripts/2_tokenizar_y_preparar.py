import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

# Cargar los textos y etiquetas
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")
y_test = pd.read_csv("data/y_test.csv")

# Cargar el tokenizer de BERT base uncased
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Función para tokenizar los textos
def tokenize_texts(texts):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

# Tokenizar
print("🔄 Tokenizando textos de entrenamiento...")
train_encodings = tokenize_texts(X_train.values.flatten())
print("✅ Entrenamiento tokenizado.")

print("🔄 Tokenizando textos de test...")
test_encodings = tokenize_texts(X_test.values.flatten())
print("✅ Test tokenizado.")

# Guardar los tensores y etiquetas para usar en el modelo
import torch

train_labels = torch.tensor(y_train.values.flatten())
test_labels = torch.tensor(y_test.values.flatten())

# Guardar todo en un archivo para cargar rápido después
torch.save({
    'train_encodings': train_encodings,
    'train_labels': train_labels,
    'test_encodings': test_encodings,
    'test_labels': test_labels
}, "data/dataset_tokenizado.pt")

print("✅ Dataset tokenizado guardado en data/dataset_tokenizado.pt")

