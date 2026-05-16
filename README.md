# Football Analysis ⚽

Sistema de análisis predictivo de fútbol con **OCI (Opción D)** + **AWS (Opción A)**.
Validado con los 64 partidos del **Mundial de Qatar 2022**.

## Resultados 🏆

| Modelo | Precisión (Qatar 2022) |
|--------|----------------------|
| 🥇 **XGBoost** | **54.7%** |
| 🥈 **MLP (Neural Network)** | **53.1%** |
| 🥉 **Logistic Regression** | **45.3%** |
| *Baseline aleatorio* | *33.3%* |

### Feature más importante
**elo_diff** — Diferencia de rating Elo entre equipos (32.5% de importancia en XGBoost)

## Arquitectura

```
┌─────────────────────┐     ┌──────────────────────┐
│   Opción D (OCI)     │     │  Opción A (AWS)      │
│  oracle-standard3    │     │  dev-vps (t3.large)   │
│  VM.Standard3.Flex   │     │                       │
│  4 OCPU · 64GB RAM   │     │  Exposición de datos  │
│  Entrenamiento ML    │     │  Validación cruzada   │
│  Pipeline de datos   │     │  APIs ligeras         │
└──────────┬───────────┘     └──────────────────────┘
           │
           └──────────► GitHub (código + resultados)
```

- **Opción D** (OCI): Entrenamiento de modelos, feature engineering, procesamiento masivo
- **Opción A** (AWS): Servir resultados, validación, endpoints
- **Sin integración** con el servidor local de OpenClaw
- **Fase 3 (LLM/RAG) omitida** por disponibilidad de GPUs

## Estructura del repo

```
football-analysis/
├── data/
│   ├── raw/              # Datasets originales (Kaggle 1872-2024)
│   └── processed/        # Datos limpios con features
├── cloud_scripts/        # Scripts de pipeline y entrenamiento
│   ├── data_pipeline.py  # Descarga, limpieza, feature engineering
│   ├── train_models.py   # Entrenamiento (XGBoost, MLP, Logistic)
│   └── run_pipeline.py   # Orchestrador (--data-only, --train-only)
├── notebooks/
│   └── validation_qatar_2022.ipynb  # Validación y visualizaciones
├── models/               # Modelos entrenados (.pkl)
├── docs/                 # Documentación
├── requirements.txt
└── README.md
```

## Setup rápido

```bash
# Clonar
git clone https://github.com/thehackerman777/football-analysis.git
cd football-analysis

# Virtual env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### En OCI

```bash
# Pipeline completo
python3 cloud_scripts/run_pipeline.py

# Solo datos
python3 cloud_scripts/run_pipeline.py --data-only

# Solo entrenamiento (requiere datos procesados)
python3 cloud_scripts/run_pipeline.py --train-only
```

## Datos

- **47,399** partidos internacionales (1872-2024)
- **64** partidos de Qatar 2022 para validación
- **9** features: elo_home, elo_away, elo_diff, form_home, form_away, form_diff, is_neutral, is_worldcup, year

## Próximas mejoras

- [ ] Incorporar datos de cuotas de apuestas en tiempo real
- [ ] TSI avanzados + PCA para reducción dimensional
- [ ] Mejorar predicción de empates (class weighting)
- [ ] Opción A: API ligera para servir predicciones
- [ ] DVC para versionado de datasets

## Fuentes

- [Global Football Results (1872-2024)](https://www.kaggle.com/datasets/muhammadehsan02/global-football-results-18722024)
- [Construction of 2022 Qatar World Cup match result prediction model](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2024.1410632/full)
