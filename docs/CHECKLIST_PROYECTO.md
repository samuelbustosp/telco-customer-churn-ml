# Checklist Operativo: Proyecto Final Machine Learning (Telco Churn)

Este checklist detalla cada actividad técnica y analítica que debes realizar de principio a fin. Todas las etapas han sido auditadas, ejecutadas y validadas con código reproducible en el notebook y módulos asociados.

---

## 🛠️ Fase 0: Entorno y Estructura Inicial
- [x] **Configuración del entorno de trabajo:**
  - [x] Crear y activar entorno virtual (`venv` o `conda`).
  - [x] Instalar librerías base: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `shap`, `jupyter`.
  - [x] Generar archivo `requirements.txt`.
- [x] **Estructura del repositorio:**
  - [x] Crear carpeta `notebooks/` y ubicar el notebook principal `telco_customer_churn.ipynb`.
  - [x] Crear carpeta `dataset/` con el archivo `Telco-Customer-Churn.csv` y su diccionario en `dataset/README.md`.
  - [x] Configurar `.gitignore` para ignorar `.ipynb_checkpoints/`, `__pycache__/`, entornos virtuales, etc.
> 🔍 **Punto de Auditoría 0:** Revisión de estructura, dependencias y reproducibilidad del entorno. *Aprobado.*

---

## 📊 Fase 1: Carga, Auditoría y Análisis Exploratorio de Datos (EDA)
- [x] **Carga e inspección inicial:**
  - [x] Cargar el CSV y verificar dimensiones (7.043 filas, 21 columnas).
  - [x] Identificar tipos de datos (`dtypes`).
  - [x] Detectar y corregir la anomalía de `TotalCharges` (conversión de texto a numérico y tratamiento de los 11 valores vacíos con `tenure == 0` imputados en 0.0).
  - [x] Descartar `customerID` para el análisis y modelado.
- [x] **Análisis de la Variable Objetivo (`Churn`):**
  - [x] Calcular frecuencias absolutas y relativas (balance 73,46% No vs 26,54% Yes).
  - [x] Visualizar la distribución de clases (gráfico de barras y gráfico de dona).
- [x] **Análisis Univariado y Descriptivo:**
  - [x] Variables numéricas (`tenure`, `MonthlyCharges`, `TotalCharges`): estadísticas de resumen (`describe()`), skewness, kurtosis, histogramas KDE y boxplots.
  - [x] Variables categóricas: conteo de frecuencias y cardinalidad de cada servicio, tipo de contrato y método de pago con porcentajes anotados.
- [x] **Análisis Bivariado / Multivariado frente a `Churn`:**
  - [x] Churn según tipo de contrato (*Month-to-month* 42,7% vs *1/2 years* 11,3% y 2,8%).
  - [x] Churn según antigüedad (*tenure*): mediana de 10 meses en fuga vs 38 meses en clientes retenidos.
  - [x] Churn según servicios contratados (*TechSupport*, *OnlineSecurity* duplican retención).
  - [x] Churn según método de pago (cheque electrónico alcanza 45,3% de fuga) y cargos mensuales altos (\$70 a \$105).
  - [x] Matriz de correlación de Pearson entre variables numéricas y Churn.
- [x] **Síntesis de Hallazgos:**
  - [x] Redactar síntesis de hallazgos clave del EDA que fundamenten las decisiones de preprocesamiento.

> 🔍 **Punto de Auditoría 1:** Limpieza de datos fundamentada, 4 hipótesis validadas empíricamente y tabla de síntesis metodológica completada. *Aprobado.*

---

## ⚙️ Fase 2: Preprocesamiento e Ingeniería de Atributos (*Feature Engineering*)
- [x] **Ingeniería de Atributos:**
  - [x] Crear variable de servicios acumulados (`total_addon_services`).
  - [x] Crear variable de ratio de cargos (`monthly_to_total_ratio`).
  - [x] Crear cohortes de antigüedad (`tenure_group`: 0-12m, 12-24m, 24-48m, >48m).
  - [x] Crear indicador de cliente sin cargas familiares (`is_alone`).
  - [x] Crear indicador de vulnerabilidad contractual (`is_month_to_month`).
- [x] **Partición de Datos (¡antes de codificar y escalar para evitar *Data Leakage*!):**
  - [x] Separar matriz de características ($X$) y vector objetivo ($y$).
  - [x] Dividir en entrenamiento y prueba (`train_test_split`, 80/20) con estratificación (`stratify=y`) y semilla fija (`random_state=42`).
- [x] **Transformaciones y Pipelines:**
  - [x] Codificación One-Hot de variables categóricas (`drop='first'`, `handle_unknown='ignore'`).
  - [x] Escalado de variables numéricas (`StandardScaler`).
  - [x] Integrar transformaciones mediante `ColumnTransformer` en `src/preprocessing.py` y el notebook principal.

> 🔍 **Punto de Auditoría 2:** Prevención total de *data leakage*, pipeline scikit-learn modular y reproducibilidad certificada. *Aprobado.*

---

## 🤖 Fase 3: Modelado y Línea Base (*Baseline*)
- [x] **Modelo de Referencia (*Baseline*):**
  - [x] Entrenar Regresión Logística con `class_weight='balanced'`.
  - [x] Evaluar métricas iniciales con validación cruzada estratificada (`StratifiedKFold`, 5 folds).
- [x] **Modelos Candidatos:**
  - [x] Árbol de Decisión (`DecisionTreeClassifier`).
  - [x] Random Forest (`RandomForestClassifier`).
  - [x] Boosting (`XGBClassifier` con `scale_pos_weight`).
- [x] **Manejo del Desbalance de Clases:**
  - [x] Ponderación balanceada de clases en todos los clasificadores.
- [x] **Tabla Comparativa Inicial:**
  - [x] Comparar ROC-AUC (0.846 en RF y XGBoost), PR-AUC, Recall (>78-80%), Precision y F1 en CV.

> 🔍 **Punto de Auditoría 3:** Evaluación de métricas en 5-folds, estabilidad estadística confirmada y selección de XGBoost y Random Forest para optimización. *Aprobado.*

---

## 🎯 Fase 4: Optimización de Hiperparámetros
- [x] **Definición de Grillas y Espacios de Búsqueda:**
  - [x] Definir grillas de hiperparámetros para XGBoost (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`) y Random Forest (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`).
- [x] **Búsqueda Sistemática:**
  - [x] Ejecutar `GridSearchCV` optimizando sobre `scoring='roc_auc'` con 5-fold CV.
  - [x] Registrar mejores hiperparámetros y puntuaciones medias obtenidas.
- [x] **Evaluación del Sobreajuste (*Overfitting*):**
  - [x] Comparar desempeño en entrenamiento vs validación (brecha controlada entre 0.04 y 0.05).

> 🔍 **Punto de Auditoría 4:** Hiperparámetros óptimos identificados, sobreajuste controlado y tiempo de convergencia balanceado. *Aprobado.*

---

## 📈 Fase 5: Evaluación Final, Selección y Ajuste de Umbral
- [x] **Evaluación en Conjunto de Test:**
  - [x] Evaluar los modelos finales optimizados en el conjunto de prueba retenido (1.409 clientes).
  - [x] Reportar métricas: ROC-AUC (0.847 en XGBoost), PR-AUC (0.663), Recall, Precision, F1-Score y Accuracy.
  - [x] Generar Matriz de Confusión y Curvas duales ROC / Precision-Recall.
- [x] **Optimización de Umbral de Decisión (*Threshold Tuning*):**
  - [x] Analizar variación de Recall, Precision y F1 al mover el umbral de 0,1 a 0,9.
  - [x] Seleccionar y justificar el umbral óptimo orientado al negocio (`threshold = 0.35`), elevando el **Recall al 89,3%** (334 clientes en fuga captados sobre 374).
- [x] **Justificación de Selección del Modelo:**
  - [x] Justificación técnica y comercial documentada a favor de XGBoost.

> 🔍 **Punto de Auditoría 5:** Generalización en test validada, curva de costo-beneficio analizada y umbral 0.35 fundamentado para retención proactiva. *Aprobado.*

---

## 🧠 Fase 6: Explicabilidad e Interpretabilidad con SHAP
- [x] **Configuración de SHAP:**
  - [x] Inicializar `shap.TreeExplainer` sobre el clasificador XGBoost optimizado.
  - [x] Calcular valores SHAP sobre el conjunto de test.
- [x] **Explicabilidad Global:**
  - [x] Generar gráfico *Summary Plot* (Beeswarm) con magnitud y dirección del impacto de cada atributo.
  - [x] Generar gráfico de barras de importancia media absoluta ($E[|SHAP|]$).
  - [x] Analizar gráficos de dependencia (*Dependence Plots*) en `Contract`, `tenure` y `MonthlyCharges`.
- [x] **Explicabilidad Local:**
  - [x] Generar *Waterfall Plot* para un cliente de alto riesgo de abandono ($P > 80\%$).
  - [x] Generar *Waterfall Plot* para un cliente fidelizado de bajo riesgo ($P < 8\%$).
- [x] **Traducción a Estrategia de Negocio:**
  - [x] Conclusiones claras sobre las palancas comerciales que determinan la retención.

> 🔍 **Punto de Auditoría 6:** Explicabilidad global y local plenamente consistente con la intuición de negocio. *Aprobado.*

---

## 📝 Fase 7: Conclusiones, Notebook Final y Repositorio
- [x] **Redacción en el Notebook:**
  - [x] Conclusiones ejecutivas vinculadas a las hipótesis iniciales.
  - [x] Plan de acción comercial estructurado en 4 pilares estratégicos.
  - [x] Limitaciones del modelo y recomendaciones de trabajo futuro (monitoreo de drift, CLV).
- [x] **Limpieza del Notebook:**
  - [x] Ejecución secuencial limpia y sin errores de punta a punta (74 celdas, 33 de código ejecutadas con outputs).
  - [x] Sin warnings innecesarios ni celdas de prueba residuales.
- [x] **Repositorio y Documentación:**
  - [x] `README.md` completo con contexto, estructura, resumen de resultados e instrucciones de reproducción.
  - [x] Módulos de producción limpios y documentados en `src/preprocessing.py` y `src/evaluation.py`.
  - [x] `dataset/README.md` con diccionario de datos exhaustivo.
  - [x] `requirements.txt` con todas las dependencias necesarias.

> 🔍 **Punto de Auditoría 7:** Entrega final completada y verificada de punta a punta. *Aprobado para entrega.*
