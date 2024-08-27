import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

# Cargar los datasets de entrenamiento y prueba
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# Configuración de la aplicación Streamlit
st.title("Predicción Multiclase del Riesgo de Obesidad")
st.write("Este conjunto de datos se genera a partir de un modelo de aprendizaje profundo entrenado con el conjunto de datos de riesgo de obesidad o CVD. Las distribuciones de características son similares a las del conjunto de datos original, lo que lo hace ideal para visualizaciones y análisis exploratorio de datos.")
st.write("""
**Dataset**

El Dataset utilizado contiene información sobre diversos factores demográficos, comportamentales y físicos que influyen en la obesidad. Cada registro en el conjunto de datos representa a un individuo e incluye las siguientes características:

- **Género**: Masculino, Femenino
- **Edad**: Edad en años
- **Altura**: Altura en metros
- **Peso**: Peso en kilogramos
- **Historial Familiar con Sobrepeso**: sí, no
- **Consumo Frecuente de Alimentos Altamente Calóricos (FAVC)**: sí, no
- **Frecuencia de Actividad Física (CAEC)**: A veces, Frecuentemente, no, Siempre
- **Consumo de Vegetales (SCC)**: sí, no
- **Número de Comidas por Día**: Valor numérico
- **Consumo de Agua por Día**: Valor numérico en litros
- **Hábito de Fumar (SMOKE)**: sí, no
- **Consumo Diario de Alcohol (CALC)**: A veces, no, Frecuentemente
- **Modo de Transporte (MTRANS)**: Transporte Público, Automóvil, Caminando, Motocicleta, Bicicleta
- **Nivel de Riesgo de Obesidad (NObeyesdad)**: Sobrepeso Nivel II, Peso Normal, Peso Insuficiente, Obesidad Tipo III, Obesidad Tipo II, Sobrepeso Nivel I, Obesidad Tipo I
""")

# Información básica sobre los datasets
st.header("Información de los Datasets")
st.write(f'Tamaño del dataset de entrenamiento: {df_train.shape}')
st.dataframe(df_train.head(3))
st.write(f'Tamaño del dataset de prueba: {df_test.shape}')
st.dataframe(df_test.head(3))

# Visualización de la distribución de características numéricas
st.header("Distribución de Características Numéricas")
numerical_features = df_train.select_dtypes(include=['int64', 'float64']).columns[1:]

fig, ax = plt.subplots(figsize=(16, 12))
df_train[numerical_features].hist(ax=ax, bins=20)
plt.tight_layout()
st.pyplot(fig)

# Visualización de la distribución de características categóricas
st.header("Distribución de Características Categóricas")
categorical_features = df_train.select_dtypes(include=['object']).columns
num_features = len(categorical_features)
num_rows = (num_features + 1) // 2 
num_cols = min(2, num_features)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 6*num_rows))
axes = axes.flatten()

for i, col in enumerate(categorical_features):  
    counts = df_train[col].value_counts()
    wedges, texts, autotexts = axes[i].pie(counts, autopct='%1.1f%%', startangle=90)
    for text in texts + autotexts:
        text.set_fontsize(10)

    axes[i].set_title(f'Distribution of {col}')
    axes[i].legend(wedges, counts.index, title=col, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
st.pyplot(fig)

# Visualización de características categóricas vs la variable objetivo
st.header("Distribución de Características Categóricas vs Variable Objetivo")
fig, axes = plt.subplots(5, 2, figsize=(16, 24))
axes = axes.flatten()
for i, col in enumerate(categorical_features):
    sns.countplot(x=col, hue='NObeyesdad', data=df_train, ax=axes[i])
    axes[i].set_title(f'{col} vs NObeyesdad')
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Preparación de los datos para el modelo
train = df_train.drop("id", axis=1) 
test = df_test.drop("id", axis=1)    

categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = train.select_dtypes(include=['float64', 'int64']).columns.tolist()

categorical_cols.remove('NObeyesdad')  # 'NObeyesdad' es la variable objetivo

# Codificación de las características categóricas
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]], axis=0))  
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# Codificación de la variable objetivo
le = LabelEncoder()
train['NObeyesdad'] = le.fit_transform(train['NObeyesdad'])

# Separación de características y variable objetivo
X = train.drop("NObeyesdad", axis=1)
y = train["NObeyesdad"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)

# Visualización de los resultados del modelo
st.header("Resultados del Modelo")
st.write(f"Precisión del modelo: {accuracy}")

labels = le.classes_
fig, ax = plt.subplots(figsize=(15, 7))
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.xticks(rotation=45, fontsize=10)
ax.set_title("Matriz de Confusión")
st.pyplot(fig)

# Predicción en el conjunto de prueba
test_pred = model.predict(test)
test_pred_labels = le.inverse_transform(test_pred)

submission = pd.DataFrame({
    'id': df_test['id'],
    'NObeyesdad': test_pred_labels
})

st.success("Predicción creada exitosamente!")
st.dataframe(submission)
