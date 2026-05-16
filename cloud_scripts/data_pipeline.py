#!/usr/bin/env python3
"""
Pipeline de datos para análisis predictivo de fútbol.
Ejecuta: descarga → limpieza → feature engineering → export.
Corre en OCI (Opción D).

Fuente: Kaggle - Global Football Results (1872-2026)
"""

import os, sys, json, warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.environ.get('DATA_DIR', '/opt/football-analysis/data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROC_DIR = os.path.join(DATA_DIR, 'processed')
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

# ─── 1. Descarga ─────────────────────────────────────────────────
def download_dataset():
    """Descarga el dataset de resultados internacionales desde Kaggle."""
    import kagglehub
    print("[1/5] Descargando dataset de Kaggle...")
    path = kagglehub.dataset_download("muhammadehsan02/global-football-results-18722024")
    import shutil
    for f in os.listdir(path):
        shutil.copy(os.path.join(path, f), RAW_DIR)
        print(f"  → {f}")
    print(f"  Destino: {RAW_DIR}")
    return os.listdir(RAW_DIR)

# ─── 2. Carga y limpieza ─────────────────────────────────────────
def load_and_clean():
    """Carga results.csv, normaliza nombres históricos, crea variables base."""
    import pandas as pd
    print("[2/5] Cargando y limpiando datos...")
    
    # Buscar el archivo results
    csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv') and 'result' in f.lower()]
    if not csv_files:
        # Intentar cualquier CSV
        csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No se encontró archivo CSV en datos crudos")
    
    filepath = os.path.join(RAW_DIR, csv_files[0])
    print(f"  Archivo: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"  Registros crudos: {len(df):,}")
    print(f"  Columnas: {list(df.columns)}")
    print(f"  Rango fechas: {df['date'].min()} → {df['date'].max()}")
    
    # Normalizar nombres históricos
    country_mapping = {
        'Soviet Union': 'Russia',
        'West Germany': 'Germany',
        'East Germany': 'Germany',
        'Czechoslovakia': 'Czech Republic',
        'Yugoslavia': 'Serbia',
        'Zaire': 'DR Congo',
        'Netherlands Antilles': 'Curacao',
    }
    for col in ['home_team', 'away_team']:
        df[col] = df[col].replace(country_mapping)
    
    # Variable objetivo
    df['total_goals'] = df['home_score'] + df['away_score']
    df['result'] = df.apply(
        lambda x: 'H' if x['home_score'] > x['away_score']
        else ('A' if x['away_score'] > x['home_score'] else 'D'),
        axis=1
    )
    
    # Parsear fecha
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    
    # Crear flag de torneo
    tournament_keywords = ['FIFA World Cup', 'World Cup', 'UEFA Euro', 'Copa America',
                           'African Cup', 'Asian Cup', 'CONCACAF', 'Oceania']
    df['is_tournament'] = df['tournament'].apply(
        lambda x: any(kw.lower() in str(x).lower() for kw in tournament_keywords)
    )
    
    print(f"  Registros limpios: {len(df):,}")
    print(f"  Distribución resultados: H={sum(df['result']=='H')}, "
          f"A={sum(df['result']=='A')}, D={sum(df['result']=='D')}")
    
    return df

# ─── 3. Feature Engineering ──────────────────────────────────────
def engineer_features(df):
    """Construye variables predictoras: Elo, forma reciente, neutral, ranking proxy."""
    import numpy as np
    import pandas as pd
    print("[3/5] Ingeniería de características...")
    
    # Ordenar por fecha
    df = df.sort_values('date').reset_index(drop=True)
    
    # Elo rating simple
    teams = {}
    elo_home, elo_away = [], []
    
    for i, row in df.iterrows():
        home, away = row['home_team'], row['away_team']
        if home not in teams:
            teams[home] = 1500.0
        if away not in teams:
            teams[away] = 1500.0
        
        elo_home.append(teams[home])
        elo_away.append(teams[away])
        
        # Actualizar Elo post-partido
        expected_h = 1 / (1 + 10 ** ((teams[away] - teams[home]) / 400))
        actual_h = 1.0 if row['result'] == 'H' else (0.5 if row['result'] == 'D' else 0.0)
        K = 20 if row['tournament'] == 'Friendly' else 32
        
        teams[home] += K * (actual_h - expected_h)
        teams[away] += K * (expected_h - actual_h)  # 1 - actual_h for away view
    
    df['elo_home'] = elo_home
    df['elo_away'] = elo_away
    df['elo_diff'] = df['elo_home'] - df['elo_away']
    
    # Forma reciente (últimos 5 partidos)
    from collections import defaultdict, deque
    recent_form = defaultdict(lambda: deque(maxlen=5))
    form_features = []
    
    for i, row in df.iterrows():
        home, away = row['home_team'], row['away_team']
        
        home_form = sum(recent_form[home]) / 5 if recent_form[home] else 0.5
        away_form = sum(recent_form[away]) / 5 if recent_form[away] else 0.5
        form_features.append((home_form, away_form))
        
        # Actualizar forma (1=win local, 0.5=draw, 0=loss local)
        home_points = 1.0 if row['result'] == 'H' else (0.5 if row['result'] == 'D' else 0.0)
        away_points = 1.0 if row['result'] == 'A' else (0.5 if row['result'] == 'D' else 0.0)
        recent_form[home].append(home_points)
        recent_form[away].append(away_points)
    
    df['form_home'], df['form_away'] = zip(*form_features)
    df['form_diff'] = df['form_home'] - df['form_away']
    
    # Neutral venue
    if 'neutral' in df.columns:
        df['is_neutral'] = df['neutral'].astype(int)
    else:
        df['is_neutral'] = 0
    
    # Equipos FIFA World Cup (para filtrado)
    df['is_worldcup'] = df['tournament'].str.contains('FIFA World Cup', na=False).astype(int)
    
    print(f"  Features creadas: elo_diff, form_diff, is_neutral, is_worldcup")
    print(f"  Rango Elo: {df['elo_diff'].min():.0f} a {df['elo_diff'].max():.0f}")
    
    return df

# ─── 4. Filtrar Mundial 2022 para validación ────────────────────
def extract_qatar2022(df):
    """Extrae los partidos de Qatar 2022 para validación."""
    import pandas as pd
    print("[4/5] Extrayendo Mundial Qatar 2022...")
    
    qatar = df[(df['tournament'] == 'FIFA World Cup') & (df['year'] == 2022)].copy()
    print(f"  Partidos encontrados: {len(qatar)}")
    
    if len(qatar) == 0:
        print("  ⚠️ No se encontraron partidos. Buscando variantes...")
        qatar = df[df['tournament'].str.contains('World Cup', na=False) & (df['year'] == 2022)].copy()
        print(f"  Con variante: {len(qatar)} partidos")
    
    if len(qatar) > 0:
        # Los IDs de grupo en el dataset original
        print(f"  Fechas: {qatar['date'].min()} → {qatar['date'].max()}")
        print(f"  Equipos: {sorted(set(qatar['home_team'].unique()) | set(qatar['away_team'].unique()))}")
    
    return qatar

# ─── 5. Exportar ────────────────────────────────────────────────
def export_data(df, qatar_df):
    """Exporta datasets procesados."""
    import pandas as pd
    print("[5/5] Exportando datos procesados...")
    
    # Full dataset con features
    out_full = os.path.join(PROC_DIR, 'matches_with_features.csv')
    df.to_csv(out_full, index=False)
    print(f"  Full features: {out_full} ({len(df):,} registros)")
    
    # Qatar 2022
    out_qatar = os.path.join(PROC_DIR, 'qatar_2022.csv')
    qatar_df.to_csv(out_qatar, index=False)
    print(f"  Qatar 2022: {out_qatar} ({len(qatar_df)} registros)")
    
    # Resumen de features
    feature_cols = [c for c in df.columns if c not in 
                    ['date', 'home_team', 'away_team', 'tournament', 'city', 'country',
                     'neutral', 'home_score', 'away_score', 'result']]
    
    summary = {
        'total_matches': len(df),
        'date_range': [str(df['date'].min()), str(df['date'].max())],
        'features': feature_cols,
        'target_distribution': df['result'].value_counts().to_dict(),
        'qatar_matches': len(qatar_df),
        'unique_teams': df['home_team'].nunique(),
    }
    
    with open(os.path.join(PROC_DIR, 'dataset_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Resumen: dataset_summary.json")
    
    return summary

# ─── MAIN ────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("FOOTBALL DATA PIPELINE — Opción D (OCI)")
    print("=" * 60)
    
    files = download_dataset()
    df = load_and_clean()
    df = engineer_features(df)
    qatar_df = extract_qatar2022(df)
    summary = export_data(df, qatar_df)
    
    print("\n✅ Pipeline completado exitosamente")
    print(f"   Total partidos: {summary['total_matches']:,}")
    print(f"   Features: {len(summary['features'])}")
    print(f"   Qatar 2022: {summary['qatar_matches']} partidos")
