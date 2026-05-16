# Football Analysis ⚽

Sistema de análisis predictivo de fútbol con **OCI (Opción D)** + **AWS (Opción A)**.
Validado con los 64 partidos del **Mundial de Qatar 2022**.

## 🔗 Web App
**http://18.213.174.229/** — Tier list + predicciones interactivas

## Resultados 🏆

| Modelo | Dataset | Precisión |
|--------|---------|-----------|
| 🥇 **Ensemble** (Hist + TSI) | 47k hist + 250 TSI | **64.1%** (41/64) |
| 🥈 **XGBoost TSI** | 250 partidos con eventos | **59.4%** (38/64) |
| 🥉 **XGBoost Original** | 47k resultados históricos | **54.7%** (35/64) |
| MLP Neural Network | 250 TSI | 54.7% (35/64) |
| Logistic Regression | 47k históricos | 45.3% (29/64) |
| *Baseline aleatorio* | — | *33.3%* |

## 🏆 Tier List — Qatar 2022

| Tier | Equipos |
|------|---------|
| 👑 **S** | Argentina 🇦🇷, France 🇫🇷 |
| 🏅 **A** | Croatia 🇭🇷, Morocco 🇲🇦 |
| 🏆 **B** | Netherlands 🇳🇱, England 🏴󠁧󠁢󠁥󠁮󠁧󠁿, Brazil 🇧🇷, Portugal 🇵🇹 |
| ⚔️ **C** | Japan 🇯🇵, Australia 🇦🇺, Senegal 🇸🇳, Switzerland 🇨🇭, Spain 🇪🇸, USA 🇺🇸, Poland 🇵🇱, South Korea 🇰🇷 |
| 📋 **D** | Alemania, Ecuador, Camerún, Uruguay, Túnez, México, Bélgica, Ghana, Arabia Saudita, Irán, Costa Rica, Dinamarca, Serbia, Gales, Canadá, Qatar |

## Arquitectura

```
┌──────────────────────┐     ┌───────────────────────┐     ┌──────────┐
│  Opción D (OCI)       │     │  Opción A (AWS)       │     │  GitHub  │
│  oracle-standard3     │     │  dev-vps              │     │          │
│  4 OCPU · 64GB RAM    │     │  Web App (port 80)    │     │  Código  │
│  Pipeline datos       │     │  Tier list interactiva│     │  Modelos │
│  TSI de StatsBomb     │     │  Predicciones en vivo │     │  Datos   │
│  Entrenamiento ML     │     │  API REST             │     │  Docs    │
└──────────┬───────────┘     └───────────────────────┘     └──────────┘
           │
           └──────────► Clima (Open-Meteo) + StatsBomb (eventos)
```

## Features utilizadas

| Grupo | Features | Cantidad |
|-------|----------|----------|
| **Históricas** | Elo, forma reciente, H2H, promedios de gol, rendimiento en eliminatorias, historial en Mundiales | 23 |
| **TSI (eventos)** | Tiros, tiros a puerta, posesión, pases, presión, recuperaciones, intercepciones, faltas, tarjetas, córners | 39 |
| **Total** | — | **62** |
| **Datos de entrenamiento** | 47k resultados históricos + 250 partidos con eventos + clima | — |

## Próximos pasos 🚀

### Qatar 2026
- [ ] Squad data de clasificatorias (equipos probables)
- [ ] Datos de Transfermarkt (valores de mercado)
- [ ] API-Football para datos frescos
- [ ] Weather data para entrenamiento (1940-presente gratis)
- [ ] Datos de lesiones y sanciones
- [ ] Modelo específico para fase de grupos vs eliminatorias
- [ ] Simulaciones Monte Carlo para el torneo completo

### Mejoras al modelo
- [ ] Más datos de entrenamiento con TSI (StatsBomb tiene Mundiales desde 1958)
- [ ] PCA sobre TSI para reducir dimensionalidad
- [ ] XGBoost con calibración de probabilidades
- [ ] Red Neuronal con atención sobre secuencia de partidos

## Fuentes de datos
- [Kaggle: Global Football Results 1872-2024](https://www.kaggle.com/datasets/muhammadehsan02/global-football-results-18722024)
- [StatsBomb Open Data](https://github.com/statsbomb/open-data) — Eventos detallados de partidos
- [Open-Meteo](https://open-meteo.com/) — Clima histórico gratuito
- [API-Football](https://www.api-football.com/) — Datos en tiempo real
