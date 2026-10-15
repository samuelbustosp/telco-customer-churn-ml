"""
Módulo de Preprocesamiento e Ingeniería de Atributos
Proyecto Final: Telco Customer Churn
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset Telco Customer Churn:
    1. Convierte TotalCharges a float imputando espacios en blanco por 0.0 (clientes con tenure == 0).
    2. Elimina la columna customerID si está presente.
    
    Parameters:
        df_raw (pd.DataFrame): DataFrame original.
        
    Returns:
        pd.DataFrame: DataFrame limpio.
    """
    df = df_raw.copy()
    
    # 1. Corrección de TotalCharges
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce').fillna(0.0)
    
    # 2. Descarte de identificador único
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    return df


def engineer_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables derivadas con alto valor predictivo:
    - is_alone: 1 si el cliente no tiene pareja ni dependientes, 0 caso contrario.
    - total_addon_services: Conteo de servicios de valor agregado contratados.
    - tenure_group: Segmentación en cohortes de antigüedad ('0-12m', '12-24m', '24-48m', '>48m').
    - monthly_to_total_ratio: Ratio entre MonthlyCharges y TotalCharges.
    - is_month_to_month: Indicador binario si el contrato es mes a mes.
    
    Parameters:
        df_input (pd.DataFrame): DataFrame limpio.
        
    Returns:
        pd.DataFrame: DataFrame con nuevas características.
    """
    df = df_input.copy()
    
    # 1. Cliente sin cargas familiares (is_alone)
    if 'Partner' in df.columns and 'Dependents' in df.columns:
        df['is_alone'] = ((df['Partner'] == 'No') & (df['Dependents'] == 'No')).astype(int)
    
    # 2. Conteo de servicios adicionales contratados
    addon_services = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    present_addons = [col for col in addon_services if col in df.columns]
    if present_addons:
        df['total_addon_services'] = df[present_addons].apply(lambda row: sum(row == 'Yes'), axis=1)
        
    # 3. Cohortes de antigüedad (tenure_group)
    if 'tenure' in df.columns:
        bins = [-1, 12, 24, 48, 72]
        labels = ['0-12m', '12-24m', '24-48m', '>48m']
        df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels).astype(str)
        
    # 4. Ratio financiero: MonthlyCharges respecto a TotalCharges acumulado (+1 para evitar división por 0)
    if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
        df['monthly_to_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1.0)
        
    # 5. Indicador directo de contrato mes a mes
    if 'Contract' in df.columns:
        df['is_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)
        
    return df


def get_feature_lists(df: pd.DataFrame, target_col: str = 'Churn'):
    """
    Identifica y separa columnas numéricas y categóricas del DataFrame (excluyendo la variable objetivo).
    
    Parameters:
        df (pd.DataFrame): DataFrame con características.
        target_col (str): Nombre de la variable objetivo.
        
    Returns:
        tuple: (list_numeric_features, list_categorical_features)
    """
    features = [c for c in df.columns if c != target_col]
    numeric_features = df[features].select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_features = df[features].select_dtypes(include=['object', 'category']).columns.tolist()
    
    return numeric_features, categorical_features


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Construye el ColumnTransformer de scikit-learn con estandarización para numéricas
    y OneHotEncoding para categóricas evitando multicolinealidad con drop='first'.
    
    Parameters:
        numeric_features (list): Lista de características numéricas.
        categorical_features (list): Lista de características categóricas.
        
    Returns:
        ColumnTransformer: Preprocesador reproducible.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor


def split_data(df: pd.DataFrame, target_col: str = 'Churn', test_size: float = 0.2, random_state: int = 42):
    """
    Separa las características (X) y el objetivo (y), codificando Churn binariamente (0 = No, 1 = Yes),
    y divide en conjuntos de entrenamiento y prueba mediante muestreo estratificado.
    
    Parameters:
        df (pd.DataFrame): DataFrame procesado.
        target_col (str): Nombre de la columna objetivo.
        test_size (float): Proporción del conjunto de prueba (default: 0.2).
        random_state (int): Semilla aleatoria (default: 42).
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_col])
    y = (df[target_col] == 'Yes').astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test
