from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# ── Load models at startup ────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load SARIMA
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAXResults
    sarima_model = SARIMAXResults.load(os.path.join(MODELS_DIR, 'sarima_model.pkl'))
    print("SARIMA model loaded.")
except Exception as e:
    sarima_model = None
    print(f"SARIMA load error: {e}")

# Load GAM
try:
    gam_model = joblib.load(os.path.join(MODELS_DIR, 'gam_model.pkl'))
    print("GAM model loaded.")
except Exception as e:
    gam_model = None
    print(f"GAM load error: {e}")

# Load XGBoost
try:
    xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
    print("XGBoost model loaded.")
except Exception as e:
    xgb_model = None
    print(f"XGBoost load error: {e}")

# Load historical data
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'tarkwa_rainfall_clean.csv')
df_history = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)
df_history.index.freq = 'MS'

# ── Rainfall interpretation ───────────────────────────────
def interpret_rainfall(mm: float) -> dict:
    if mm <= 50:
        return {"category": "Very Low", "color": "#90CAF9",
                "message": "Very low rainfall expected. Drought risk is high. Farmers should consider irrigation."}
    elif mm <= 100:
        return {"category": "Low", "color": "#42A5F5",
                "message": "Low rainfall expected. Minimal flooding risk. Good conditions for most mining operations."}
    elif mm <= 150:
        return {"category": "Moderate", "color": "#1976D2",
                "message": "Moderate rainfall expected. Normal conditions for Tarkwa. Standard precautions apply."}
    elif mm <= 250:
        return {"category": "High", "color": "#F9A825",
                "message": "High rainfall expected. Monitor drainage systems. Farmers should prepare for good yields."}
    elif mm <= 350:
        return {"category": "Very High", "color": "#EF6C00",
                "message": "Very high rainfall expected. Flood risk is elevated. Mining operations should inspect pit drainage."}
    else:
        return {"category": "Extreme", "color": "#B71C1C",
                "message": "Extreme rainfall expected. High flood risk. Disaster management agencies should be on alert."}

# ── FastAPI app ───────────────────────────────────────────
app = FastAPI(
    title="Tarkwa Rainfall Prediction API",
    description="API for predicting monthly rainfall in Tarkwa, Ghana",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response models ───────────────────────────────
class PredictionRequest(BaseModel):
    year: int
    month: int
    model: str = "SARIMA"

class PredictionResponse(BaseModel):
    year: int
    month: int
    model: str
    predicted_rainfall_mm: float
    category: str
    color: str
    message: str
    month_name: str

# ── Helper: month names ───────────────────────────────────
MONTH_NAMES = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]

# ── Endpoints ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Tarkwa Rainfall Prediction API is running.",
            "models_available": ["SARIMA", "GAM", "XGBoost"]}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "sarima": sarima_model is not None,
        "gam": gam_model is not None,
        "xgboost": xgb_model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    year = request.year
    month = request.month
    model_name = request.model.upper()

    if month < 1 or month > 12:
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12.")
    if year < 1996 or year > 2100:
        raise HTTPException(status_code=400, detail="Year must be between 1996 and 2100.")

    predicted_mm = None

    # ── SARIMA prediction ──────────────────────────────────
    if model_name == "SARIMA":
        if sarima_model is None:
            raise HTTPException(status_code=500, detail="SARIMA model not loaded.")
        try:
            target_date = pd.Timestamp(year=year, month=month, day=1)
            last_date = df_history.index[-1]
            steps = (target_date.year - last_date.year) * 12 + \
                    (target_date.month - last_date.month)
            if steps <= 0:
                steps = 1
            forecast = sarima_model.forecast(steps=steps)
            predicted_mm = float(forecast.iloc[-1])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"SARIMA prediction error: {e}")

    # ── GAM prediction ─────────────────────────────────────
    elif model_name == "GAM":
        if gam_model is None:
            raise HTTPException(status_code=500, detail="GAM model not loaded.")
        try:
            time_index = (year - 1996) * 12 + (month - 1)
            X_input = np.array([[month, year, time_index]])
            predicted_mm = float(gam_model.predict(X_input)[0])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"GAM prediction error: {e}")

    # ── XGBoost prediction ─────────────────────────────────
    elif model_name == "XGBOOST":
        if xgb_model is None:
            raise HTTPException(status_code=500, detail="XGBoost model not loaded.")
        try:
            rainfall_series = df_history['Monthly_Rainfall'].values
            target_date = pd.Timestamp(year=year, month=month, day=1)
            last_date = df_history.index[-1]

            if target_date <= last_date:
                idx = df_history.index.get_loc(target_date)
                recent = rainfall_series[max(0, idx-12):idx]
            else:
                recent = rainfall_series[-12:]

            while len(recent) < 12:
                recent = np.concatenate([[recent[0]], recent])

            lag1   = recent[-1]
            lag2   = recent[-2]
            lag3   = recent[-3]
            lag6   = recent[-6]
            lag12  = recent[-12]
            roll3  = np.mean(recent[-3:])
            roll6  = np.mean(recent[-6:])
            roll12 = np.mean(recent[-12:])
            time_index = (year - 1996) * 12 + (month - 1)

            X_input = np.array([[month, year, time_index,
                                  lag1, lag2, lag3, lag6, lag12,
                                  roll3, roll6, roll12]])
            predicted_mm = float(xgb_model.predict(X_input)[0])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"XGBoost prediction error: {e}")

    else:
        raise HTTPException(status_code=400,
                            detail="Model must be one of: SARIMA, GAM, XGBoost.")

    predicted_mm = max(0.0, predicted_mm)
    interpretation = interpret_rainfall(predicted_mm)

    return PredictionResponse(
        year=year,
        month=month,
        model=request.model,
        predicted_rainfall_mm=round(predicted_mm, 2),
        category=interpretation["category"],
        color=interpretation["color"],
        message=interpretation["message"],
        month_name=MONTH_NAMES[month - 1]
    )

@app.get("/history")
def get_history():
    records = []
    for date, row in df_history.iterrows():
        records.append({
            "date": str(date.date()),
            "year": date.year,
            "month": date.month,
            "month_name": MONTH_NAMES[date.month - 1],
            "rainfall_mm": round(float(row['Monthly_Rainfall']), 2),
            "category": interpret_rainfall(float(row['Monthly_Rainfall']))["category"]
        })
    return {"data": records, "total_records": len(records)}

@app.get("/metrics")
def get_metrics():
    return {
        "models": [
            {"model": "SARIMA", "rmse": 71.20, "mae": 54.73,
             "mape": 126.76, "r2": 0.5667},
            {"model": "GAM",    "rmse": 80.03, "mae": 59.51,
             "mape": 132.56, "r2": 0.4527},
            {"model": "XGBoost","rmse": 84.76, "mae": 61.71,
             "mape": 190.44, "r2": 0.3892},
            {"model": "LSTM",   "rmse": 93.82, "mae": 71.75,
             "mape": 267.79, "r2": 0.2518}
        ],
        "best_model": "SARIMA",
        "note": "LSTM is available locally only due to deployment constraints."
    }