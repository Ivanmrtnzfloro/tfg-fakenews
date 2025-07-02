import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Cargar datos
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

# Añadir etiquetas: 1 = fake, 0 = real
fake_df['label'] = 1
true_df['label'] = 0

# Simular que es un post de red social combinando título y texto
fake_df['post'] = fake_df['title'].fillna('') + ". " + fake_df['text'].fillna('')
true_df['post'] = true_df['title'].fillna('') + ". " + true_df['text'].fillna('')

# Quedarnos con la columna relevante
fake_df = fake_df[['post', 'label']]
true_df = true_df[['post', 'label']]

# Unimos y mezclamos
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# División en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(
    df['post'], df['label'], test_size=0.2, random_state=42
)

# Guardar los conjuntos en archivos CSV
os.makedirs("data", exist_ok=True)
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("✅ Preprocesamiento completado. Datos listos para BERT.")

