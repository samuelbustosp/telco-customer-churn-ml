# Diccionario de Datos y Metadatos: Telco Customer Churn

Este directorio contiene el dataset oficial utilizado en el proyecto de Machine Learning para la predicción del abandono de clientes (*churn*).

## 📄 Metadatos del Dataset
* **Nombre:** Telco Customer Churn
* **Fuente:** Repositorio público de IBM Cognos Analytics / Kaggle
* **Dimensiones:** 7.043 filas y 21 columnas originales
* **Formato:** Archivo delimitado por comas (`.csv`, codificación UTF-8)
* **Variable Objetivo:** `Churn` (Binaria: `'Yes'` / `'No'`)

---

## 📋 Diccionario de Variables

| # | Columna | Tipo de Dato | Descripción de Negocio | Valores Posibles / Rango |
|:---|:---|:---|:---|:---|
| 1 | `customerID` | Categórica (str) | Identificador alfanumérico único por cliente | Texto alfanumérico (ej. `'7590-VHVEG'`) *(Descartado en modelado)* |
| 2 | `gender` | Categórica (str) | Género del cliente | `'Female'`, `'Male'` |
| 3 | `SeniorCitizen` | Numérica / Binaria | Indica si el cliente es adulto mayor (tercera edad) | `0` = No, `1` = Sí |
| 4 | `Partner` | Categórica (str) | Indica si el cliente tiene pareja | `'Yes'`, `'No'` |
| 5 | `Dependents` | Categórica (str) | Indica si el cliente tiene dependientes económicos | `'Yes'`, `'No'` |
| 6 | `tenure` | Numérica (int64) | Cantidad de meses de permanencia en la empresa | Rango: `0` a `72` meses |
| 7 | `PhoneService` | Categórica (str) | Indica si tiene contratado servicio de telefonía | `'Yes'`, `'No'` |
| 8 | `MultipleLines` | Categórica (str) | Indica si dispone de múltiples líneas telefónicas | `'No'`, `'Yes'`, `'No phone service'` |
| 9 | `InternetService` | Categórica (str) | Tipo de proveedor y tecnología de conexión a internet | `'DSL'`, `'Fiber optic'`, `'No'` |
| 10 | `OnlineSecurity` | Categórica (str) | Servicio adicional de seguridad en línea | `'No'`, `'Yes'`, `'No internet service'` |
| 11 | `OnlineBackup` | Categórica (str) | Servicio adicional de copia de seguridad en la nube | `'No'`, `'Yes'`, `'No internet service'` |
| 12 | `DeviceProtection` | Categórica (str) | Plan de protección y garantía de dispositivos | `'No'`, `'Yes'`, `'No internet service'` |
| 13 | `TechSupport` | Categórica (str) | Servicio adicional de soporte técnico prioritario | `'No'`, `'Yes'`, `'No internet service'` |
| 14 | `StreamingTV` | Categórica (str) | Servicio de transmisión de televisión en línea | `'No'`, `'Yes'`, `'No internet service'` |
| 15 | `StreamingMovies` | Categórica (str) | Servicio de transmisión de películas en línea | `'No'`, `'Yes'`, `'No internet service'` |
| 16 | `Contract` | Categórica (str) | Plazo y modalidad del contrato de servicio | `'Month-to-month'`, `'One year'`, `'Two year'` |
| 17 | `PaperlessBilling` | Categórica (str) | Modalidad de facturación electrónica sin papel | `'Yes'`, `'No'` |
| 18 | `PaymentMethod` | Categórica (str) | Medio de pago utilizado para el cobro del servicio | `'Electronic check'`, `'Mailed check'`, `'Bank transfer (automatic)'`, `'Credit card (automatic)'` |
| 19 | `MonthlyCharges` | Numérica (float64) | Monto facturado mensualmente al cliente | Rango: `$18.25` a `$118.75` |
| 20 | `TotalCharges` | Numérica (float64)* | Monto total acumulado facturado durante la permanencia | Rango: `$0.00` a `$8,684.80` *(Nota: convertida de texto a float; 11 registros con espacios se imputaron a 0.0 por tener tenure = 0)* |
| 21 | `Churn` | Categórica (str) | **Variable Objetivo:** ¿Abandonó el cliente la compañía? | `'No'` (73,46%), `'Yes'` (26,54%) |

---

## 🛠️ Variables Derivadas en el Pipeline (*Feature Engineering*)

Durante la Fase 2 del proyecto se crean las siguientes características a partir del dataset original:
- `is_alone`: Indicador binario si el cliente no tiene pareja ni dependientes (`Partner == 'No' & Dependents == 'No'`).
- `total_addon_services`: Conteo total de servicios de valor agregado contratados (0 a 6).
- `tenure_group`: Segmentación en cohortes de permanencia (`'0-12m'`, `'12-24m'`, `'24-48m'`, `'>48m'`).
- `monthly_to_total_ratio`: Relación entre el costo mensual y el monto acumulado total.
- `is_month_to_month`: Indicador binario de contrato de renovación mensual.
