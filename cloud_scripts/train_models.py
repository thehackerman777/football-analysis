#!/usr/bin/env python3
"""
Entrenamiento de modelos predictivos de fútbol.
Evalúa: Logistic Regression, XGBoost, MLP (Red Neuronal).
Validación primaria: Mundial Qatar 2022.

Corre en OCI (Opción D).
"""

import os, sys, json, warnings, pickle
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import xgboost as xgb

DATA_DIR = os.environ.get('DATA_DIR', '/opt/football-analysis/data')
PROC_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(DATA_DIR, '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

def load_data():
    """Carga datos procesados con features."""
    print("[1/6] Cargando datos procesados...")
    filepath = os.path.join(PROC_DIR, 'matches_with_features.csv')
    df = pd.read_csv(filepath, parse_dates=['date'])
    print(f"  Registros: {len(df):,}")
    return df

def prepare_train_test(df):
    """
    Prepara datos de entrenamiento.
    Usa datos pre-2022 para entrenar, Qatar 2022 para test.
    """
    print("[2/6] Preparando train/test split...")
    
    # Feature columns (excluir no-numéricas y leaks)
    exclude = ['date', 'home_team', 'away_team', 'tournament', 'city', 'country',
               'neutral', 'home_score', 'away_score', 'result', 'total_goals']
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['int64', 'float64']]
    
    print(f"  Features: {len(feature_cols)}")
    
    X = df[feature_cols].fillna(0)
    y = df['result'].values  # H, A, D
    
    # Split: pre-2022 para train, 2022 para test (Qatar World Cup)
    train_mask = df['year'] < 2022
    test_mask = (df['tournament'] == 'FIFA World Cup') & (df['year'] == 2022)
    
    # Fallback si no encuentra exactamente el torneo
    if test_mask.sum() == 0:
        print("  ⚠️ No se encontraron partidos de World Cup 2022, usando año >= 2022")
        test_mask = df['year'] >= 2022
        train_mask = ~test_mask
    
    X_train = X[train_mask]
    y_train = pd.Series(y[train_mask])
    X_test = X[test_mask]
    y_test = pd.Series(y[test_mask])
    
    print(f"  Train: {len(X_train):,} partidos")
    print(f"  Test:  {len(X_test)} partidos")
    print(f"  Test distribución: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test, feature_cols

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Regresión Logística — modelo base simple."""
    print("\n[3/6] Logistic Regression...")
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, multi_class='multinomial',
                                    class_weight='balanced', random_state=SEED))
    ])
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"  Precisión: {acc:.1%}")
    report = classification_report(y_test, y_pred, output_dict=True)
    for cls in ['H', 'A', 'D']:
        if cls in report:
            print(f"  {cls}: precision={report[cls]['precision']:.1%}, "
                  f"recall={report[cls]['recall']:.1%}, f1={report[cls]['f1-score']:.1%}")
    
    return pipe, acc, y_pred

def train_xgboost(X_train, X_test, y_train, y_test):
    """XGBoost — maneja datos tabulares y missing values eficientemente."""
    print("\n[4/6] XGBoost...")
    
    # Label encode
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train_enc,
              eval_set=[(X_test, y_test_enc)],
              verbose=False)
    
    y_pred_enc = model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"  Precisión: {acc:.1%}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"  Top 5 features:")
    for _, row in importance.head(5).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    return model, acc, y_pred, le, importance

def train_mlp(X_train, X_test, y_train, y_test):
    """MLP (Red Neuronal) — captura relaciones no lineales."""
    print("\n[5/6] Red Neuronal (MLP)...")
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=SEED,
            verbose=False
        ))
    ])
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"  Precisión: {acc:.1%}")
    report = classification_report(y_test, y_pred, output_dict=True)
    for cls in ['H', 'A', 'D']:
        if cls in report:
            print(f"  {cls}: precision={report[cls]['precision']:.1%}, "
                  f"recall={report[cls]['recall']:.1%}, f1={report[cls]['f1-score']:.1%}")
    
    return pipe, acc, y_pred

def save_results(results, feature_cols):
    """Guarda modelos y resultados."""
    print("\n[6/6] Guardando resultados...")
    
    summary = {
        'models': {},
        'feature_count': len(feature_cols),
        'features': feature_cols,
    }
    
    for name, (model, acc, y_pred, *extra) in results.items():
        model_path = os.path.join(MODELS_DIR, f'{name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        summary['models'][name] = {
            'accuracy': round(acc, 4),
            'model_file': model_path,
        }
        print(f"  {name}: {acc:.1%} → {model_path}")
    
    with open(os.path.join(MODELS_DIR, 'results_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print leaderboard
    print("\n" + "=" * 50)
    print("🏆 LEADERBOARD — Qatar 2022")
    print("=" * 50)
    sorted_models = sorted(results.items(), key=lambda x: x[1][1], reverse=True)
    for i, (name, (_, acc, _)) in enumerate(sorted_models, 1):
        print(f"  {i}. {name:30s} {acc:.1%}")

if __name__ == '__main__':
    print("=" * 60)
    print("FOOTBALL MODEL TRAINING — Opción D (OCI)")
    print("=" * 60)
    
    df = load_data()
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test(df)
    
    results = {}
    
    lr_model, lr_acc, lr_pred = train_logistic_regression(X_train, X_test, y_train, y_test)
    results['logistic_regression'] = (lr_model, lr_acc, lr_pred)
    
    xgb_model, xgb_acc, xgb_pred, xgb_le, xgb_imp = train_xgboost(X_train, X_test, y_train, y_test)
    results['xgboost'] = (xgb_model, xgb_acc, xgb_pred)
    
    mlp_model, mlp_acc, mlp_pred = train_mlp(X_train, X_test, y_train, y_test)
    results['mlp_neural_network'] = (mlp_model, mlp_acc, mlp_pred)
    
    save_results(results, feature_cols)
    
    print("\n✅ Entrenamiento completado")
