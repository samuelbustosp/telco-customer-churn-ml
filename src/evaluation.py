"""
Módulo de Evaluación y Diagnóstico de Modelos
Proyecto Final: Telco Customer Churn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)


def evaluate_cv_models(models_dict: dict, X, y, cv_folds: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    Ejecuta validación cruzada estratificada sobre múltiples modelos candidatos
    evaluando ROC-AUC, PR-AUC, F1, Recall, Precision y Accuracy.
    
    Parameters:
        models_dict (dict): Diccionario con formato {'nombre_modelo': modelo_o_pipeline}.
        X: Matriz de características (DataFrame o array).
        y: Vector objetivo binario.
        cv_folds (int): Número de particiones (default: 5).
        random_state (int): Semilla para StratifiedKFold.
        
    Returns:
        pd.DataFrame: Tabla resumen con media y desvío estándar de cada métrica.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    scoring = {
        'roc_auc': 'roc_auc',
        'pr_auc': 'average_precision',
        'f1': 'f1',
        'recall': 'recall',
        'precision': 'precision',
        'accuracy': 'accuracy'
    }
    
    results = []
    
    for name, model in models_dict.items():
        scores = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1, return_train_score=False)
        results.append({
            'Modelo': name,
            'ROC-AUC Media': scores['test_roc_auc'].mean(),
            'ROC-AUC Std': scores['test_roc_auc'].std(),
            'PR-AUC Media': scores['test_pr_auc'].mean(),
            'PR-AUC Std': scores['test_pr_auc'].std(),
            'F1 Media': scores['test_f1'].mean(),
            'F1 Std': scores['test_f1'].std(),
            'Recall Media': scores['test_recall'].mean(),
            'Recall Std': scores['test_recall'].std(),
            'Precision Media': scores['test_precision'].mean(),
            'Precision Std': scores['test_precision'].std(),
            'Accuracy Media': scores['test_accuracy'].mean(),
            'Accuracy Std': scores['test_accuracy'].std(),
        })
        
    df_results = pd.DataFrame(results).sort_values(by='ROC-AUC Media', ascending=False).reset_index(drop=True)
    return df_results


def evaluate_test_set(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """
    Calcula las métricas de rendimiento en el conjunto de prueba para un umbral determinado.
    
    Parameters:
        model: Modelo o pipeline entrenado con método predict_proba.
        X_test: Características de test.
        y_test: Vector objetivo real de test.
        threshold (float): Umbral de corte de probabilidad (default: 0.5).
        
    Returns:
        dict: Métricas cuantitativas, probabilidades y predicciones binarizadas.
    """
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
        
    y_pred = (y_proba >= threshold).astype(int)
    
    metrics = {
        'threshold': threshold,
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba),
        'f1': f1_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred),
        'y_proba': y_proba,
        'y_pred': y_pred
    }
    return metrics


def plot_confusion_matrix_custom(y_true, y_pred, title: str = 'Matriz de Confusión', ax=None):
    """
    Grafica la matriz de confusión con anotaciones absolutas y porcentuales.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    labels = np.asarray([
        [f'{count:,}\n({pct:.1%})' for count, pct in zip(row_count, row_pct)]
        for row_count, row_pct in zip(cm, cm_norm)
    ])
    
    sns.heatmap(
        cm, annot=labels, fmt='', cmap='Blues', cbar=False, ax=ax,
        xticklabels=['No Churn (0)', 'Churn (1)'],
        yticklabels=['No Churn (0)', 'Churn (1)'],
        annot_kws={'fontsize': 11, 'fontweight': 'bold'}
    )
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Predicción del Modelo', fontsize=11)
    ax.set_ylabel('Valor Real Observado', fontsize=11)
    return ax


def plot_roc_and_pr_curves(y_true, y_proba, model_name: str = 'Modelo Ganador', axes=None):
    """
    Grafica las curvas ROC y Precision-Recall con áreas bajo la curva calculadas.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        
    # 1. Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    axes[0].plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'{model_name} (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Clasificador Aleatorio (AUC = 0.500)')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=11)
    axes[0].set_ylabel('Tasa de Verdaderos Positivos (Recall)', fontsize=11)
    axes[0].set_title('Curva ROC (Receiver Operating Characteristic)', fontsize=12, fontweight='bold', pad=10)
    axes[0].legend(loc='lower right', frameon=True)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Curva Precision-Recall
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    baseline_pr = y_true.sum() / len(y_true)
    axes[1].plot(recalls, precisions, color='#d95f02', lw=2.5, label=f'{model_name} (PR-AUC = {pr_auc:.3f})')
    axes[1].axhline(y=baseline_pr, color='gray', lw=1.5, linestyle='--', label=f'Línea Base Sin Habilidad ({baseline_pr:.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall (Sensibilidad)', fontsize=11)
    axes[1].set_ylabel('Precision (Valor Predictivo Positivo)', fontsize=11)
    axes[1].set_title('Curva Precision-Recall', fontsize=12, fontweight='bold', pad=10)
    axes[1].legend(loc='upper right', frameon=True)
    axes[1].grid(True, alpha=0.3)
    
    return axes


def evaluate_threshold_range(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    """
    Evalúa el desempeño del modelo variando el umbral de decisión desde 0.05 a 0.95.
    
    Parameters:
        y_true: Valores reales (0 o 1).
        y_proba: Probabilidades estimadas para la clase 1.
        thresholds: Lista o array de umbrales a evaluar.
        
    Returns:
        pd.DataFrame: Métricas (Recall, Precision, F1, Accuracy, Falsos Positivos, Falsos Negativos) por umbral.
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)
        
    rows = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        
        rows.append({
            'Umbral': round(t, 2),
            'Recall': recall_score(y_true, preds, zero_division=0),
            'Precision': precision_score(y_true, preds, zero_division=0),
            'F1-Score': f1_score(y_true, preds, zero_division=0),
            'Accuracy': accuracy_score(y_true, preds),
            'TP (Detectados)': tp,
            'FP (Falsa Alarma)': fp,
            'FN (Fuga No Detectada)': fn,
            'TN (Retenidos Correctos)': tn
        })
        
    return pd.DataFrame(rows)
