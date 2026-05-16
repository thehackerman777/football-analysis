"""
Football Predictor — Web UI (Opción A: AWS)
Qatar 2022 Validation with betting-style interface.
"""

import json, os, re
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(title="Football Predictor — Qatar 2022")

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "qatar2022_matches.json"
BETS_FILE = BASE_DIR / "data" / "user_bets.json"
STATS_FILE = BASE_DIR / "data" / "user_stats.json"
HTML_FILE = BASE_DIR / "templates" / "index.html"

# ─── Data Loading ────────────────────────────────────────────────
def load_matches():
    with open(DATA_FILE) as f:
        return json.load(f)

def load_bets():
    if not BETS_FILE.exists():
        return {}
    with open(BETS_FILE) as f:
        return json.load(f)

def save_bets(bets):
    with open(BETS_FILE, "w") as f:
        json.dump(bets, f, indent=2)

def load_stats():
    default = {"total_bets": 0, "correct": 0, "incorrect": 0}
    if not STATS_FILE.exists():
        return default
    with open(STATS_FILE) as f:
        return json.load(f)

def save_stats(stats):
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

# ─── Routes ─────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home():
    html = HTML_FILE.read_text()
    return HTMLResponse(html)

@app.get("/api/matches")
async def api_matches():
    return load_matches()

@app.get("/api/stats")
async def api_stats():
    bets = load_bets()
    matches = load_matches()
    
    total = len(bets)
    correct = sum(1 for m in matches 
                  if bets.get(str(m["date"] + m["home"] + m["away"])) == m["result"])
    
    # Model stats
    models = {
        "xgboost": {"correct": 0, "total": len(matches)},
        "mlp_neural_network": {"correct": 0, "total": len(matches)},
        "logistic_regression": {"correct": 0, "total": len(matches)},
    }
    for m in matches:
        if m.get("xgboost_pred") == m["result"]:
            models["xgboost"]["correct"] += 1
        if m.get("mlp_neural_network_pred") == m["result"]:
            models["mlp_neural_network"]["correct"] += 1
        if m.get("logistic_regression_pred") == m["result"]:
            models["logistic_regression"]["correct"] += 1
    
    for name, data in models.items():
        data["pct"] = round(data["correct"] / data["total"] * 100, 1)
    
    return {
        "user": {
            "total": total,
            "correct": correct,
            "pct": round(correct/total*100, 1) if total > 0 else 0
        },
        "models": models,
        "total_matches": len(matches)
    }

@app.post("/api/bet")
async def place_bet(request: Request):
    data = await request.json()
    match_key = data.get("match_key")
    prediction = data.get("prediction")
    
    if not match_key or prediction not in ["H", "A", "D"]:
        raise HTTPException(400, "Invalid request")
    
    matches = load_matches()
    bets = load_bets()
    stats = load_stats()
    
    # Find match
    match = next((m for m in matches 
                  if str(m["date"] + m["home"] + m["away"]) == match_key), None)
    if not match:
        raise HTTPException(404, "Match not found")
    
    bets[match_key] = prediction
    save_bets(bets)
    
    # Recalc stats
    correct = sum(1 for m in matches 
                  if bets.get(str(m["date"] + m["home"] + m["away"])) == m["result"])
    stats["total_bets"] = len(bets)
    stats["correct"] = correct
    stats["incorrect"] = len(bets) - correct
    save_stats(stats)
    
    return {
        "status": "ok",
        "is_correct": prediction == match["result"],
        "actual": match["result"],
        "stats": stats
    }

@app.post("/api/reset")
async def reset_bets():
    if BETS_FILE.exists():
        BETS_FILE.unlink()
    save_stats({"total_bets": 0, "correct": 0, "incorrect": 0})
    return {"status": "reset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
