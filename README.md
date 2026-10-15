# 📡 Predicción de Churn en Clientes de Telecomunicaciones

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-red)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)
![Status](https://img.shields.io/badge/Status-Entrega%20Final%20Completada-success)

Proyecto integral de Machine Learning enfocado en la detección temprana del abandono de clientes (*Customer Churn*) en una compañía de telecomunicaciones. Mediante modelos probabilísticos supervisados, calibración de umbrales comerciales y técnicas de interpretabilidad con SHAP, el sistema permite focalizar los esfuerzos de retención y proteger los ingresos recurrentes de la empresa.

---

## 🎯 Contexto y Preguntas del Proyecto

El costo de adquisición de un nuevo cliente (*CAC*) en telecomunicaciones suele ser entre 5 y 10 veces superior al costo de retención de un cliente existente. Una detección predictiva a tiempo permite activar incentivos y programas de fidelización antes de que se formalice la baja.

* **Pregunta de Negocio:** *¿Es posible identificar con antelación a aquellos clientes con probabilidad crítica de rescindir sus contratos para activar acciones preventivas de retención antes de que abandonen la compañía?*
* **Pregunta Técnica:** *¿Qué algoritmo de clasificación supervisada maximiza la detección de clientes en riesgo (ROC-AUC / Recall) minimizando falsos positivos y ofreciendo explicabilidad interpretable para el equipo comercial?*

---

## 📊 Dataset y Auditoría de Datos

Se utiliza el dataset oficial **Telco Customer Churn** compuesto por **7.043 observaciones** y **21 columnas**:
* **Demografía:** Género, tercera edad (`SeniorCitizen`), pareja (`Partner`), personas a cargo (`Dependents`).
* **Servicios Contratados:** Telefonía, múltiples líneas, internet (DSL, Fibra óptica, Sin servicio), seguridad online, respaldo en nube, protección de dispositivos, soporte técnico, streaming de TV y películas.
* **Información Contractual y de Cuenta:** Plazo (`Contract`: Month-to-month, 1 año, 2 años), facturación electrónica (`PaperlessBilling`), método de pago (`PaymentMethod`) y antigüedad (`tenure` en meses).
* **Variables Financieras:** Cargos mensuales (`MonthlyCharges`) y acumulados (`TotalCharges`).
* **Variable Objetivo (`Churn`):**
  * `No`: 5.174 clientes (73,46%) — *Cliente activo / fidelizado*
  * `Yes`: 1.869 clientes (26,54%) — *Cliente que abandonó el servicio*

> 🔍 **Auditoría de Calidad:** Se detectaron 11 registros con espacios en blanco en `TotalCharges`, todos correspondientes a clientes con `tenure == 0`. Fueron imputados en `$0.00` de forma fundamentada sin pérdida de registros. Se descartó la columna `customerID` por carecer de poder predictivo.

---

## 📁 Estructura del Repositorio

```text
telco-customer-churn-ml/
│
├── README.md                          <- Informe ejecutivo, metodología y resultados finales
├── requirements.txt                   <- Dependencias fijadas del entorno virtual
├── .gitignore                         <- Reglas de exclusión de Git
│
├── dataset/
│   ├── Telco-Customer-Churn.csv       <- Dataset oficial del proyecto
│   └── README.md                      <- Diccionario de variables y metadatos
│
├── docs/                              <- Marco metodológico y académico
│   ├── INFORME_FINAL_PROYECTO.docx    <- Informe final académico y ejecutivo en formato Word (.docx)
│   ├── INFORME_FINAL_PROYECTO.md      <- Informe final completo en formato Markdown
│   ├── CONSIGNA.md                    <- Guía oficial de requisitos de la entrega
│   ├── CONTEXTO_Y_OBJETIVOS.md        <- Formulación técnica, hipótesis y preguntas clave
│   ├── CHECKLIST_PROYECTO.md          <- Checklist de auditoría y cumplimiento de hitos
│   └── figures/                       <- Gráficos de alta resolución (EDA, modelos, curvas y SHAP)
│
├── notebooks/
│   └── telco_customer_churn.ipynb     <- Notebook principal reproducible end-to-end (Fases 1 a 7)
│
└── src/                               <- Módulos reutilizables de producción
    ├── __init__.py
    ├── preprocessing.py               <- Limpieza, feature engineering y ColumnTransformer
    └── evaluation.py                  <- Validación cruzada, curvas ROC/PR y threshold tuning
```

---

## 🔬 Metodología y Fases del Ciclo de Vida ML

### Fase 1: Análisis Exploratorio de Datos (EDA) y Validación de Hipótesis
* **H1 (Contractual) [CONFIRMADA]:** Clientes mes a mes tienen una tasa de abandono del **42,71%**, frente al **11,27%** en contratos a 1 año y apenas **2,83%** a 2 años.
* **H2 (Antigüedad) [CONFIRMADA]:** La mediana de antigüedad de los clientes en fuga es de sólo **10 meses** (vs. 38 meses en retenidos); el riesgo crítico se concentra en $tenure \le 12$ meses.
* **H3 (Servicios) [CONFIRMADA]:** Clientes sin `TechSupport` u `OnlineSecurity` registran >**41%** de abandono vs. **~15%** cuando cuentan con el servicio.
* **H4 (Financiera) [CONFIRMADA]:** El pago por `Electronic check` presenta una tasa de churn del **45,3%** (el triple de las transferencias automáticas con ~15%), y las fugas aumentan fuertemente en planes de \$70 a \$105/mes.

### Fase 2: Preprocesamiento e Ingeniería de Atributos
* **Nuevas variables derivadas:**
  * `is_alone`: Cliente sin pareja ni dependientes.
  * `total_addon_services`: Conteo de servicios de valor agregado contratados (0 a 6).
  * `tenure_group`: Segmentación en cohortes (`0-12m`, `12-24m`, `24-48m`, `>48m`).
  * `monthly_to_total_ratio`: Relación entre cobro mensual y facturación histórica acumulada.
  * `is_month_to_month`: Indicador específico de vulnerabilidad contractual.
* **Partición y Pipeline:**
  * Partición estratificada 80/20 (`stratify=y`, `random_state=42`) previa a transformaciones para evitar *Data Leakage*.
  * `ColumnTransformer`: `StandardScaler` en numéricas y `OneHotEncoder(drop='first')` en categóricas.

### Fase 3: Modelado y Validación Cruzada
Se evaluaron 4 modelos con **5-Fold Stratified Cross-Validation** incorporando balanceo de clases:

| Modelo | ROC-AUC Media | PR-AUC Media | Recall Media | Precision Media | F1-Score Media | Accuracy Media |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | **0.846 ± 0.010** | 0.662 ± 0.017 | 78.5% ± 1.9% | 52.7% ± 1.1% | **0.631 ± 0.011** | 75.6% |
| **XGBoost** | **0.846 ± 0.012** | **0.663 ± 0.023** | **80.3% ± 2.5%** | 51.9% ± 1.3% | 0.630 ± 0.016 | 75.0% |
| **Regresión Logística (Baseline)** | 0.846 ± 0.012 | 0.664 ± 0.013 | 79.7% ± 3.3% | 51.8% ± 1.5% | 0.627 ± 0.018 | 74.9% |
| **Árbol de Decisión** | 0.827 ± 0.011 | 0.611 ± 0.025 | 78.0% ± 2.5% | 49.8% ± 0.6% | 0.608 ± 0.010 | 73.3% |

### Fase 4: Optimización de Hiperparámetros
Mediante `GridSearchCV` sobre métrica `roc_auc` se optimizaron los mejores ensambles:
* **XGBoost Optimizado:** `n_estimators=100`, `max_depth=3`, `learning_rate=0.08`, `subsample=0.8`, `colsample_bytree=0.8`.
* **Diagnóstico de Sobreajuste:** Brecha Train-Validation controlada en 0.04 - 0.05, garantizando alta capacidad de generalización.

### Fase 5: Evaluación en Test y Optimización de Umbral (*Threshold Tuning*)
Resultados finales en el conjunto de prueba retenido (1.409 clientes):

| Configuración | Umbral | ROC-AUC | PR-AUC | Recall (Sensibilidad) | Precision | F1-Score | Accuracy | Fugas Detectadas |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost Tuned (Estándar)** | `0.50` | **0.847** | **0.663** | 80.5% | 51.5% | **0.628** | 74.7% | 301 de 374 |
| **XGBoost Tuned (Negocio)** | **`0.35`** | **0.847** | **0.663** | **89.3%** | 45.1% | 0.599 | 69.8% | **334 de 374** |

> 💡 **Criterio de Negocio:** Se adopta el **umbral 0.35** para el entorno comercial. Esto incrementa el Recall a casi un **90%**, evitando que 33 clientes adicionales se fuguen inadvertidos a cambio de un costo operativo mínimo de contacto.

---

## 🧠 Interpretabilidad con SHAP

El análisis con `shap.TreeExplainer` sobre el modelo XGBoost reveló los factores causales determinantes:

1. **Tipo de Contrato (`Contract_Two year` / `Contract_One year`):** Representa el factor de mayor peso defensivo; fija anclas a largo plazo que disminuyen drásticamente los log-odds de abandono.
2. **Antigüedad del Cliente (`tenure`):** Relación monotónica inversa; cada mes de permanencia adicional reduce la propensión de abandono.
3. **Servicios de Asistencia (`TechSupport` y `OnlineSecurity`):** Los clientes con estos servicios activos presentan dependencias técnicas que multiplican la lealtad.
4. **Método de Pago (`Electronic check`):** Es el principal impulsor positivo de fuga, reflejando fricción en el cobro y falta de adhesión al servicio.
5. **Cargos Mensuales (`MonthlyCharges`):** Clientes de fibra óptica con facturas superiores a \$75/mes sin servicios adicionales tienen alta sensibilidad al precio.

---

## 📈 Recomendaciones Estratégicas para el Negocio

A partir de la evidencia estadística y las explicaciones de SHAP, se plantea una estrategia de retención estructurada en **4 Pilares de Acción**:

1. **Protocolo de Onboarding Intensivo (Primeros 12 Meses):**
   * El 50% de las bajas ocurren antes del décimo mes. Implementar contactos preventivos proactivos a los 30, 90 y 180 días con soporte técnico gratuito.
2. **Plan de Migración de Contratos Mes a Mes:**
   * Ofrecer un descuento promocional del 10% durante 3 meses a clientes en planes mensuales que acepten migrar a contratos de 1 o 2 años.
3. **Estrategia de Paquetización de Soporte (*Bundling*):**
   * Bonificar los módulos de `TechSupport` y `OnlineSecurity` durante el primer trimestre a clientes que adquieran planes de fibra óptica.
4. **Campaña de Adhesión a Débito Automático:**
   * Otorgar un descuento único o beneficio de gigabytes adicionales a clientes que cambien de cheque electrónico a tarjeta de crédito o débito automático.

---

## ⚙️ Instrucciones de Instalación y Reproducción

### 1. Clonar el repositorio
```bash
git clone https://github.com/samuelbustosp/telco-customer-churn-ml.git
cd telco-customer-churn-ml
```

### 2. Crear y activar el entorno virtual
En Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

En Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar requerimientos
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Abrir y ejecutar el notebook
```bash
jupyter notebook notebooks/telco_customer_churn.ipynb
```
*El notebook se encuentra completamente pre-ejecutado con todos los gráficos, curvas y métricas guardadas.*

---

## 📚 Documentación de Soporte
* [Consigna Académica de la Entrega](docs/CONSIGNA.md)
* [Contexto de Negocio y Objetivos Técnicos](docs/CONTEXTO_Y_OBJETIVOS.md)
* [Checklist Operativo de Auditoría](docs/CHECKLIST_PROYECTO.md)
* [Diccionario de Datos](dataset/README.md)
