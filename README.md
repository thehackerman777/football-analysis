# Football Analysis ⚽

Sistema de análisis predictivo de fútbol con **OCI (Opción D)** + **AWS (Opción A)**.
Validado con los 64 partidos del **Mundial de Qatar 2022**.

## 🔗 Web App
**http://18.213.174.229/** — Interfaz estilo betting para validar predicciones

## Resultados 🏆

| Modelo | Aciertos | Precisión |
|--------|----------|-----------|
| 🥇 **XGBoost** | 35/64 | **54.7%** |
| 🥈 **MLP (Neural Network)** | 34/64 | **53.1%** |
| 🥉 **Logistic Regression** | 29/64 | **45.3%** |
| *Baseline aleatorio* | *21/64* | *33.3%* |

### Feature más importante
**elo_diff** — Diferencia de rating Elo entre equipos (32.5% de importancia en XGBoost)

## Arquitectura

```
┌─────────────────────┐     ┌───────────────────────┐
│  Opción D (OCI)      │     │  Opción A (AWS)       │
│  oracle-standard3    │     │  dev-vps (t3.large)   │
│  VM.Standard3.Flex   │     │  Web App (port 80)    │
│  4 OCPU · 64GB RAM   │     │  UI betting interactiva│
│  Entrenamiento ML    │     │  API REST             │
│  Pipeline de datos   │     │  Systemd persistente  │
└──────────┬───────────┘     └───────────────────────┘
           │
           └──────────► GitHub
```

## Estructura

```
football-analysis/
├── data/raw/           # Datasets originales (Kaggle 1872-2024)
├── data/processed/     # 47,399 registros con features
├── cloud_scripts/      # Pipeline y entrenamiento
│   ├── data_pipeline.py
│   ├── train_models.py
│   └── run_pipeline.py
├── web-app/            # UI de validación (FastAPI + HTML/JS)
│   ├── app.py
│   ├── templates/index.html
│   └── data/qatar2022_matches.json (64 partidos)
├── notebooks/          # Jupyter Notebook de validación
├── models/             # Modelos .pkl + results_summary.json
├── deploy/             # Systemd + Nginx config
└── README.md
```

## Infraestructura

| Recurso | Proveedor | Spec | IP | Estado |
|---------|-----------|------|-----|--------|
| oracle-standard3 | OCI (Opción D) | 4 OCPU · 64GB RAM | 157.137.219.155 | ✅ Running |
| dev-vps | AWS (Opción A) | t3.large · 8GB RAM | 18.213.174.229 | ✅ Running |

## Setup

```bash
git clone https://github.com/thehackerman777/football-analysis.git
cd football-analysis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Pipeline completo
python3 cloud_scripts/run_pipeline.py

# Web app
cd web-app && uvicorn app:app --host 0.0.0.0 --port 8080
```

## Próximas mejoras 🚀

- [ ] **Incorporar squad data** (jugadores, valores de mercado, edad promedio)
- [ ] **FIFA Rankings históricos** como feature adicional
- [ ] **Datos de eliminatorias** para contextualizar forma reciente
- [ ] **Cuotas de apuestas** en tiempo real
- [ ] **TSI avanzados + PCA** (44 indicadores → componentes principales)
- [ ] **API-Football** para datos frescos
- [ ] **Modelo 2026** con datos de clasificación

## Fuentes

- [Global Football Results (1872-2024)](https://www.kaggle.com/datasets/muhammadehsan02/global-football-results-18722024)
- [Construction of 2022 Qatar World Cup match result prediction model](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2024.1410632/full)
- [Predicting football match outcomes: MLP model based on TSI](https://pmc.ncbi.nlm.nih.gov/articles/PMC12708546/)
