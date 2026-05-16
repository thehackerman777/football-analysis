# Football Analysis ⚽

Sistema de análisis predictivo de fútbol con **OCI (Opción D)** + **AWS (Opción A)**.
Validado con los 64 partidos del **Mundial de Qatar 2022**.

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
           └──────────► GitHub (código + datasets procesados)
```

- **Opción D** (OCI): Entrenamiento de modelos, feature engineering, procesamiento masivo
- **Opción A** (AWS): Servir resultados, validación, endpoints
- **Sin integración** con el servidor local de OpenClaw
- **Fase 3 (LLM/RAG) omitida** por disponibilidad de GPUs

## Modelos implementados

| Algoritmo | Precisión (Qatar 2022) |
|-----------|----------------------|
| Red Neuronal (MLP) | ~86.7% |
| XGBoost | ~58-69% |
| ANN | ~75.4% |
| Regresión Logística | ~61.3% |

## Estructura del repo

```
football-analysis/
├── data/
│   ├── raw/              # Datasets originales (Kaggle 1872-2026)
│   └── processed/        # Datos limpios y feature-engineered
├── cloud_scripts/        # Scripts para OCI/AWS Run Command
├── notebooks/            # Jupyter Notebooks
│   └── validation_qatar_2022.ipynb
├── docs/                 # Documentación
├── requirements.txt
└── README.md
```

## Setup rápido

```bash
# Clonar
git clone git@github.com:thehackerman777/football-analysis.git
cd football-analysis

# Virtual env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Fuentes de datos

- [Global Football Results (1872-2026)](https://www.kaggle.com/datasets/muhammadehsan02/global-football-results-18722024)
- API-Football (alineaciones, eventos, cuotas)
- StatsBomb Open Data (eventos granulares)
