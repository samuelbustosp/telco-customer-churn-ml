# Checklist Operativo: Proyecto Final Machine Learning (Telco Churn)

Este checklist detalla cada actividad técnica y analítica que debes realizar de principio a fin. Está diseñado para que avances paso a paso a tu propio ritmo; en cada hito podés pedirme una **auditoría, corrección de código, feedback de métricas u optimización**.

---

## 🛠️ Fase 0: Entorno y Estructura Inicial
- [x] **Configuración del entorno de trabajo:**
  - [x] Crear y activar entorno virtual (`venv` o `conda`).
  - [x] Instalar librerías base: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `shap`, `jupyter`.
  - [x] Generar archivo `requirements.txt`.
- [x] **Estructura del repositorio:**
  - [x] Crear carpeta `notebooks/` y ubicar el notebook principal `telco_customer_churn.ipynb`.
  - [x] Crear carpeta `data/` o `dataset/` con el archivo `Telco-Customer-Churn.csv`.
  - [x] Configurar `.gitignore` para ignorar `.ipynb_checkpoints/`, `__pycache__/`, entornos virtuales, etc.
> 🔍 **Punto de Auditoría 0:** Revisión de estructura, dependencias y reproducibilidad del entorno.

---

## 📊 Fase 1: Carga, Auditoría y Análisis Exploratorio de Datos (EDA)
- [ ] **Carga e inspección inicial:**
  - [ ] Cargar el CSV y verificar dimensiones (7.043 filas, 21 columnas).
  - [ ] Identificar tipos de datos (`dtypes`).
  - [ ] Detectar y corregir la anomalía de `TotalCharges` (conversión de texto a numérico y tratamiento de los 11 valores vacíos con `tenure == 0`).
  - [ ] Descartar `customerID` para el análisis y modelado.
- [ ] **Análisis de la Variable Objetivo (`Churn`):**
  - [ ] Calcular frecuencias absolutas y relativas (balance ~73,5% No vs ~26,5% Yes).
  - [ ] Visualizar la distribución de clases (gráfico de barras o dona).
- [ ] **Análisis Univariado y Descriptivo:**
  - [ ] Variables numéricas (`tenure`, `MonthlyCharges`, `TotalCharges`): estadísticas de resumen (`describe()`), histogramas y boxplots para detectar asimetrías o outliers.
  - [ ] Variables categóricas: conteo de frecuencias y cardinalidad de cada servicio, tipo de contrato y método de pago.
- [ ] **Análisis Bivariado / Multivariado frente a `Churn`:**
  - [ ] Churn según tipo de contrato (*Month-to-month* vs *1/2 years*).
  - [ ] Churn según antigüedad (*tenure*): tasas de abandono en clientes nuevos vs antiguos.
  - [ ] Churn según servicios contratados (*InternetService*, *TechSupport*, *OnlineSecurity*).
  - [ ] Churn según método de pago y cargos mensuales.
  - [ ] Matriz de correlación entre variables numéricas.
- [ ] **Síntesis de Hallazgos:**
  - [ ] Redactar 3 o 4 *insights* clave del EDA que fundamenten las decisiones de modelado.

> 🔍 **Punto de Auditoría 1:** Revisión de código de EDA, tratamiento de nulos/inconsistencias y calidad de los gráficos y conclusiones.

---

## ⚙️ Fase 2: Preprocesamiento e Ingeniería de Atributos (*Feature Engineering*)
- [ ] **Ingeniería de Atributos:**
  - [ ] Crear variable de servicios acumulados (conteo de servicios de valor agregado).
  - [ ] Crear variable de ratio o disparidad de cargos (`MonthlyCharges` vs `TotalCharges / tenure`).
  - [ ] Crear rangos/cohortes de antigüedad (*tenure groups*, ej. 0-12m, 12-24m, etc.).
  - [ ] Crear indicador de cliente sin cargas familiares (`is_alone`).
- [ ] **Partición de Datos (¡antes de codificar y escalar para evitar *Data Leakage*!):**
  - [ ] Separar matriz de características ($X$) y vector objetivo ($y$).
  - [ ] Dividir en conjuntos de entrenamiento y prueba (`train_test_split`) con estratificación (`stratify=y`) y semilla fija (`random_state=42`).
- [ ] **Transformaciones y Pipelines:**
  - [ ] Codificación de variables categóricas (One-Hot Encoding con `drop='first'` o `handle_unknown='ignore'`).
  - [ ] Escalado de variables numéricas (`StandardScaler` o `RobustScaler`) en aquellas que lo requieran (regresión logística).
  - [ ] Integrar transformaciones mediante `ColumnTransformer` y `Pipeline` de `scikit-learn` para garantizar flujo reproducible.

> 🔍 **Punto de Auditoría 2:** Validación de pipelines, prevención de data leakage y justificación de nuevas variables.

---

## 🤖 Fase 3: Modelado y Línea Base (*Baseline*)
- [ ] **Modelo de Referencia (*Baseline*):**
  - [ ] Entrenar un modelo base simple (Regresión Logística o DummyClassifier).
  - [ ] Evaluar métricas iniciales con validación cruzada estratificada (`StratifiedKFold`).
- [ ] **Modelos Candidatos:**
  - [ ] Árbol de Decisión (`DecisionTreeClassifier`).
  - [ ] Random Forest (`RandomForestClassifier`).
  - [ ] Boosting (`XGBClassifier` o `GradientBoostingClassifier` / `LGBMClassifier`).
- [ ] **Manejo del Desbalance de Clases:**
  - [ ] Evaluar el impacto de parámetros de balanceo (`class_weight='balanced'` en sklearn o `scale_pos_weight` en XGBoost).
- [ ] **Tabla Comparativa Inicial:**
  - [ ] Comparar ROC-AUC, PR-AUC, Recall, Precision y F1 en validación cruzada para cada modelo candidato con sus hiperparámetros por defecto.

> 🔍 **Punto de Auditoría 3:** Evaluación de la tabla comparativa, estabilidad de la validación cruzada y selección de modelos para optimizar.

---

## 🎯 Fase 4: Optimización de Hiperparámetros
- [ ] **Definición de Grillas y Espacios de Búsqueda:**
  - [ ] Definir grillas con sentido para los 1 o 2 mejores modelos (ej. `n_estimators`, `max_depth`, `learning_rate`, `min_samples_split`, `subsample`).
- [ ] **Búsqueda Sistemática:**
  - [ ] Ejecutar `GridSearchCV` o `RandomizedSearchCV` optimizando específicamente sobre la métrica elegida (`scoring='roc_auc'` o `'f1'`).
  - [ ] Registrar mejores hiperparámetros y puntuaciones medias obtenidas en validación cruzada.
- [ ] **Evaluación del Sobreajuste (*Overfitting*):**
  - [ ] Comparar desempeño en entrenamiento vs validación para descartar sobreajuste excesivo.

> 🔍 **Punto de Auditoría 4:** Revisión de rangos de hiperparámetros, tiempos de cómputo y control de sobreajuste.

---

## 📈 Fase 5: Evaluación Final, Selección y Ajuste de Umbral
- [ ] **Evaluación en Conjunto de Test:**
  - [ ] Evaluar el modelo final optimizado en el conjunto de prueba retenido (*test set*).
  - [ ] Reportar métricas: ROC-AUC, PR-AUC, Precision, Recall, F1-Score y Accuracy.
  - [ ] Generar la Matriz de Confusión y la Curva ROC / Curva Precision-Recall.
- [ ] **Optimización de Umbral de Decisión (*Threshold Tuning*):**
  - [ ] Analizar cómo varían el Recall y la Precision al mover el umbral de probabilidad de 0,1 a 0,9.
  - [ ] Seleccionar y justificar un umbral óptimo orientado al negocio (priorizar captar clientes en riesgo minimizando costos comerciales).
- [ ] **Justificación de Selección del Modelo:**
  - [ ] Redactar la fundamentación técnica y comercial de por qué se elige este modelo frente a las alternativas.

> 🔍 **Punto de Auditoría 5:** Validación de métricas finales en test, calibración del umbral y justificación de negocio.

---

## 🧠 Fase 6: Explicabilidad e Interpretabilidad con SHAP
- [ ] **Configuración de SHAP:**
  - [ ] Inicializar el explicador adecuado (`shap.TreeExplainer` para árboles/XGBoost o `LinearExplainer`/`KernelExplainer`).
  - [ ] Calcular los valores SHAP sobre una muestra representativa o conjunto de test.
- [ ] **Explicabilidad Global:**
  - [ ] Generar gráfico *Summary Plot* (Beeswarm) para ver magnitud y dirección del impacto de cada variable.
  - [ ] Generar gráfico de barras con la importancia media absoluta ($|SHAP|$).
  - [ ] Analizar gráficos de dependencia (*Dependence Plot*) en las variables más críticas (ej. `Contract`, `tenure`, `MonthlyCharges`).
- [ ] **Explicabilidad Local:**
  - [ ] Generar un *Waterfall Plot* o *Force Plot* para:
    - Un cliente clasificado con alto riesgo de abandono (entender por qué se va).
    - Un cliente clasificado como fidelizado/leal (entender qué lo retiene).
- [ ] **Traducción a Estrategia de Negocio:**
  - [ ] Sintetizar recomendaciones comerciales basadas en los insights de SHAP.

> 🔍 **Punto de Auditoría 6:** Interpretación correcta de los gráficos de SHAP y coherencia con los objetivos comerciales.

---

## 📝 Fase 7: Conclusiones, Notebook Final y Repositorio
- [ ] **Redacción en el Notebook:**
  - [ ] Conclusiones ejecutivas vinculadas a las hipótesis iniciales.
  - [ ] Limitaciones del modelo y recomendaciones de trabajo futuro (monitoreo, *drift*, despliegue).
- [ ] **Limpieza del Notebook:**
  - [ ] Ejecutar *Restart & Run All* para verificar ejecución secuencial limpia y sin errores.
  - [ ] Eliminar celdas de pruebas sueltas, salidas intermedias gigantescas o warnings innecesarios.
- [ ] **Repositorio y Documentación:**
  - [ ] Redactar un `README.md` completo con contexto, estructura, resumen de resultados e instrucciones de reproducción.
  - [ ] Asegurar que `requirements.txt` tenga versiones compatibles.
  - [ ] Verificar que el enlace público (GitHub o similar) funcione en modo incógnito.

> 🔍 **Punto de Auditoría 7:** Auditoría integral previa a la entrega final.
