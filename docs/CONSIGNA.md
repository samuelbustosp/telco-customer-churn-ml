# Proyecto Final de Machine Learning: Consigna de Entrega

## 1. Descripción General
El objetivo de esta entrega final es integrar todas las habilidades y conocimientos adquiridos a lo largo del curso para desarrollar y presentar un **proyecto de Machine Learning completo, profesional y reproducible**. 

El trabajo debe documentar de forma clara desde la formulación del problema de negocio hasta la justificación del modelo final seleccionado y su explicabilidad, entregándose mediante un repositorio público o enlace accesible que permita su auditoría y reproducción.

---

## 2. Marco Teórico y Fases del Proyecto

### 2.1. Formulación del Problema
- **Definición de la tarea:** Determinar si el problema corresponde a clasificación o regresión.
- **Variable objetivo ($y$):** Identificar con precisión la variable a predecir y su significado.
- **Métrica de éxito:** Definir la(s) métrica(s) principal(es) alineada(s) con el impacto de negocio (ej. ROC-AUC, PR-AUC, F1-Score para clasificación; RMSE, MAE o $R^2$ para regresión).
- **Justificación de negocio:** Explicar cómo la resolución de este problema aporta valor a la organización o proceso.

### 2.2. Análisis Exploratorio de Datos (EDA)
- **Estadística descriptiva:** Distribución de variables numéricas y frecuencias de categóricas usando librerías como `pandas` y `numpy`.
- **Calidad de datos:** Identificación y diagnóstico de valores faltantes (*missing values*), registros duplicados e inconsistencias.
- **Detección de outliers:** Análisis visual y estadístico de valores atípicos.
- **Visualización:** Generación de gráficos informativos con `matplotlib` o `seaborn` para examinar correlaciones, dispersión y relaciones bivariadas/multivariadas con la variable objetivo.

### 2.3. Ingeniería de Atributos (*Feature Engineering*)
- **Preprocesamiento:** Imputación de nulos y manejo adecuado de outliers.
- **Codificación:** Transformación de variables categóricas (One-Hot Encoding, Ordinal Encoding, Target Encoding, etc.).
- **Escalado:** Normalización o estandarización (`StandardScaler`, `MinMaxScaler`, `RobustScaler`) según la sensibilidad de los algoritmos elegidos.
- **Creación de nuevas variables:** Derivación de atributos y agregaciones que aporten mayor señal predictiva.
- **Justificación:** Toda transformación debe documentarse explicando el motivo técnico o de negocio de su aplicación.

### 2.4. Entrenamiento de Modelos
- **Partición de datos:** División estratificada o temporal en conjuntos de entrenamiento (*train*), validación (*validation*) y testeo (*test*) para evitar fuga de datos (*data leakage*).
- **Librerías recomendadas:** `scikit-learn`, `xgboost`, `lightgbm`, etc.
- **Modelos a considerar:** 
  - Modelos base / *baseline* (ej. Regresión Logística / Regresión Lineal).
  - Modelos basados en árboles y ensambles (ej. Random Forest, Gradient Boosting, XGBoost).
- **Registro de experimentos:** Documentación de parámetros iniciales y desempeño de cada modelo.

### 2.5. Optimización de Hiperparámetros
- **Técnicas de búsqueda:** Implementación de búsqueda sistemática como `GridSearchCV` o `RandomizedSearchCV` (u `Optuna`).
- **Validación Cruzada (*Cross-Validation*):** Uso de esquemas robustos (ej. *Stratified K-Fold*) para evitar sobreajuste (*overfitting*).
- **Análisis de resultados:** Comparación del rendimiento antes y después del ajuste fino.

### 2.6. Selección y Evaluación del Modelo
- **Comparación cuantitativa:** Tabla comparativa de métricas en validación/test (ROC-AUC, Precision, Recall, F1, Log-Loss / MSE, MAE, $R^2$).
- **Curvas de diagnóstico:** Matriz de confusión, Curva ROC, Curva Precision-Recall o gráfico de residuos.
- **Criterio de negocio:** Justificación del modelo ganador balanceando desempeño estadístico, interpretabilidad, costo computacional y viabilidad de puesta en producción.

### 2.7. Explicabilidad e Interpretabilidad con SHAP
- **Aplicación de SHAP (`shap`):** Generación de explicaciones globales y locales para el modelo seleccionado (ej. `TreeExplainer`).
- **Gráficos clave:**
  - *Summary plot* (Beeswarm / Bar) para evaluar la importancia y dirección del impacto de las características.
  - *Force plot* o *Waterfall plot* para casos individuales de predicción.
  - *Dependence plots* para estudiar efectos no lineales o interacciones.
- **Traducción al negocio:** Conclusiones claras sobre qué variables traccionan las decisiones del modelo.

### 2.8. Preparación del Notebook Reproducible
- Estructura limpia y secuencial en formato `.ipynb`.
- Código comentado, modular y sin errores de ejecución secuencial (*Restart & Run All*).
- Semillas fijadas (`random_state`) para asegurar consistencia en los resultados.

### 2.9. Documentación y Entrega del Repositorio
- Repositorio público (GitHub, GitLab, etc.).
- `README.md` exhaustivo con:
  - Descripción del proyecto y objetivos.
  - Estructura del repositorio.
  - Instrucciones claras para clonar y reproducir el entorno (ej. `requirements.txt` o `environment.yml`).
  - Resumen ejecutivo de los resultados y modelo final.

---

## 3. Checklist de Autoevaluación

Antes de entregar, verifica el cumplimiento de cada uno de los siguientes puntos:

- [ ] **Variable objetivo y métrica:** ¿Están claramente definidas y justificadas en función del negocio?
- [ ] **EDA:** ¿Se identificaron patrones clave, correlaciones, valores faltantes y outliers mediante visualizaciones y estadísticas?
- [ ] **Ingeniería de atributos:** ¿Las transformaciones (encoding, escalado, nuevas variables) están fundamentadas y sin *data leakage*?
- [ ] **Optimización de hiperparámetros:** ¿Se utilizó validación cruzada y métodos como `GridSearchCV` o `RandomizedSearchCV`?
- [ ] **Selección y justificación:** ¿La elección del modelo final se apoya en métricas comparativas y criterios de negocio?
- [ ] **Explicabilidad (SHAP):** ¿Se interpretó el comportamiento del modelo y la relevancia de sus variables con la librería `shap`?
- [ ] **Reproducibilidad:** ¿El notebook corre de principio a fin sin fallos y con las dependencias especificadas?
- [ ] **Repositorio público:** ¿El repositorio contiene el código, el `README.md` detallado y las instrucciones de ejecución?

---

## 4. Modalidad y Formato de Entrega

- **Canal de envío:** Link público al entregable.
- **Formatos admitidos:**
  - URL pública de repositorio (GitHub / GitLab) con notebook, README y requerimientos.
  - Documento público en Google Docs / Google Drive / Google Slides (con los enlaces correspondientes al código y resultados).
  - URL de aplicación desplegada (ej. Streamlit, HuggingFace Spaces), si aplica.
