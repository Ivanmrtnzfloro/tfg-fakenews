import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Cargar los datos originales
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

# Añadir etiquetas
fake_df["label"] = "Fake"
true_df["label"] = "Real"

# Unir ambos
df = pd.concat([fake_df, true_df], ignore_index=True)

# Contar valores
class_counts = df["label"].value_counts()

# Mostrar por consola
print("Distribución de clases:")
print(class_counts)

# Visualizar
sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="Set2")
plt.title("Distribución de clases: Fake vs Real")
plt.ylabel("Número de noticias")
plt.xlabel("Clase")
plt.tight_layout()

# Crear directorio para guardar la imagen
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/distribucion_clases.png")
plt.show()
