# 🧠 Fake News Detection with BERT

Este proyecto implementa un sistema de detección automática de noticias falsas en redes sociales utilizando modelos de lenguaje preentrenados, concretamente **BERT**, junto con herramientas de procesamiento de texto y aprendizaje profundo.

> 🔍 Trabajo Fin de Grado – Iván Martínez Floro

---

## 📁 Estructura del Proyecto

- `data/` – Archivos Fake.csv y True.csv  
- `models/` – Modelo entrenado (`modelo_epoca1.pt`)  
- `templates/` – Plantilla HTML para la interfaz web  
- `figures/` – Gráficos generados (ej. distribución de clases)  
- `1_preprocesamiento.py` – Preprocesamiento de los datos  
- `2_tokenizacion.py` – Tokenización de textos  
- `3_entrenar_modelo.py` – Entrenamiento del modelo BERT  
- `4_evaluar_modelo.py` – Evaluación del modelo entrenado  
- `5_interfaz.py` – Interfaz por consola  
- `6_API.py` – API REST con Flask e interfaz web  
- `visualizar_clases.py` – Gráfico de distribución de clases  
- `requirements.txt` – Dependencias del proyecto  
- `README.md` – Este archivo

---

## ⚙️ Requisitos

- Python 3.10+
- Pip
- Se recomienda crear un entorno virtual (`venv`)

### Instalación de dependencias

pip install -r requirements.txt

Cómo ejecutar el proyecto
DESDE DENTRO DE SCRIPTS:

## 📁 Archivos grandes excluidos


Los siguientes archivos han sido excluidos del repositorio por superar el límite de tamaño de GitHub.  
Puedes descargarlos desde este enlace de Google Drive:

https://drive.google.com/drive/folders/1IVIbdhEc3_vMLF8mrOxju4dH1S1hgYzF?usp=sharing

Se deben de poner el modelo dentro de models 
y el dataset tokenizado dentro de data

O si prefieres desde el principio estos son los pasos:

1. Preprocesamiento y tokenización

python 1_preprocesamiento.py
python 2_tokenizacion.py

2. Entrenamiento del modelo

(IMPORTANTE CREAR EL ARCHIVO MODELS)

mkdir models

python 3_entrenar_modelo.py
(CUIDADO TARDA BASTANTE EN CARGAR EL MODELO SIN UNA GPU, NO LO VOY A SUBIR DIRECTAMENTE PESA MUCHO PARA GIT)

3. Evaluación del modelo

python 4_evaluar_modelo.py


4. Interfaz por consola

python 5_interfaz.py

5. API Web con Flask

python 6_API.py

Luego abre tu navegador en:

http://127.0.0.1:5000

Visualización
Para generar y guardar un gráfico de distribución de clases:

python visualizar_clases.py

Modelo utilizado

Modelo preentrenado: bert-base-uncased de Hugging Face.

Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Licencia
Este proyecto se ha desarrollado con fines académicos y puede ser reutilizado, adaptado o ampliado libremente para otros estudios relacionados con NLP, clasificación de texto y desinformación.
