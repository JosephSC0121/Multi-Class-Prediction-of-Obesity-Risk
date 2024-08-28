# Predicción Multiclase del Riesgo de Obesidad

Esta aplicación de **Streamlit** utiliza un modelo de **XGBoost** para predecir el riesgo de obesidad basado en múltiples características de los individuos. El código incluye etapas de preprocesamiento de datos, visualización de las distribuciones de las características, entrenamiento del modelo y evaluación de su rendimiento.

## Estructura del Código

1. **Importación de Librerías**:
   - `pandas`, `matplotlib`, `seaborn`, `streamlit` para la manipulación de datos, visualización y creación de la interfaz.
   - `sklearn` para la preparación de datos y evaluación del modelo.
   - `xgboost` para el modelo de clasificación.

2. **Carga de Datos**:
   - Los datasets de entrenamiento (`train.csv`) y prueba (`test.csv`) se cargan utilizando `pandas`.

3. **Visualización de Datos**:
   - Distribución de características numéricas y categóricas.
   - Análisis comparativo entre las características categóricas y la variable objetivo (`NObeyesdad`).

4. **Preprocesamiento**:
   - Eliminación de la columna `id`.
   - Codificación de las características categóricas con `LabelEncoder`.
   - Separación de los datos en características (`X`) y la variable objetivo (`y`).
   - División del dataset en conjuntos de entrenamiento y validación.

7. **Entrenamiento del Modelo**:
   - Entrenamiento del modelo `XGBClassifier` con los datos de entrenamiento.

8. **Evaluación del Modelo**:
   - Cálculo de la precisión del modelo.
   - Visualización de la matriz de confusión.

9. **Predicción en el Conjunto de Prueba**:
   - Predicciones en el conjunto de prueba y creación de un archivo de salida con los resultados.

## Requisitos

- Python 3.x
- Librerías necesarias:
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `streamlit`
  - `scikit-learn`
  - `xgboost`

Puedes instalar las dependencias necesarias utilizando el siguiente comando:

```bash
pip install pandas matplotlib seaborn streamlit scikit-learn xgboost
```
Correr: 
```bash
streamlit run main.py
```
[Video](https://www.youtube.com/watch?v=VJgwfG208Vk)
