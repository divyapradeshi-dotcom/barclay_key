import traceback
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- MODEL LOADING ----------------

try:
    model = joblib.load("pre_delinquency_model.pkl")
    print("Model loaded successfully!")
except Exception:
    print("Model failed to load")
    print(traceback.format_exc())
    model = None


# ---------------- INPUT SCHEMA ----------------

class CustomerData(BaseModel):
    limit_bal: float
    sex: int
    education: int
    marriage: int
    age: int

    pay_0: int
    pay_2: int
    pay_3: int
    pay_4: int
    pay_5: int
    pay_6: int

    bill_amt1: float
    bill_amt2: float
    bill_amt3: float
    bill_amt4: float
    bill_amt5: float
    bill_amt6: float

    pay_amt1: float
    pay_amt2: float
    pay_amt3: float
    pay_amt4: float
    pay_amt5: float
    pay_amt6: float


# ---------------- FEATURE ENGINEERING ----------------

def compute_prediction(input_df: pd.DataFrame):

    df = input_df.copy()

    pay_cols = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
    bill_cols = ["bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6"]
    pay_amt_cols = ["pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"]

    df["avg_delay"] = df[pay_cols].mean(axis=1)
    df["delay_trend"] = df["pay_6"] - df["pay_0"]

    df["bill_growth"] = (df["bill_amt6"] - df["bill_amt1"]) / (df["bill_amt1"] + 1)

    df["utilization_avg"] = df[bill_cols].mean(axis=1) / (df["limit_bal"] + 1)

    df["pay_cover_ratio_avg"] = (
        df[pay_amt_cols].mean(axis=1) /
        (df[bill_cols].mean(axis=1) + 1)
    )

    df["cash_flow_proxy"] = df[pay_amt_cols].mean(axis=1)

    # One-hot encoding (MATCH TRAINING FEATURES)
    df["sex_2"] = (df["sex"] == 2).astype(int)

    for i in range(1, 7):
        df[f"education_{i}"] = (df["education"] == i).astype(int)

    for i in range(1, 4):
        df[f"marriage_{i}"] = (df["marriage"] == i).astype(int)

    feature_order = [
        "limit_bal",
        "age",
        "avg_delay",
        "delay_trend",
        "pay_cover_ratio_avg",
        "bill_growth",
        "utilization_avg",
        "cash_flow_proxy",
        "sex_2",
        "education_1",
        "education_2",
        "education_3",
        "education_4",
        "education_5",
        "education_6",
        "marriage_1",
        "marriage_2",
        "marriage_3",
    ]

    X = df[feature_order]

    risk_score = float(model.predict_proba(X)[0][1])

    if risk_score < 0.3:
        level = "LOW RISK"
        action = "Approve normally"
        reason = "Customer shows stable repayment behaviour"
    elif risk_score < 0.7:
        level = "MEDIUM RISK"
        action = "Approve with caution"
        reason = "Customer has moderate risk indicators"
    else:
        level = "HIGH RISK"
        action = "Manual review required"
        reason = "Customer shows strong default signals"

    return risk_score, level, action, reason


# ---------------- SINGLE PREDICTION ----------------

@app.post("/predict")
def predict(data: CustomerData):

    if model is None:
        return {"error": "Model not loaded"}

    try:
        input_df = pd.DataFrame([data.dict()])

        risk_score, level, action, reason = compute_prediction(input_df)

        return {
            "risk_score": risk_score,
            "risk_level": level,
            "recommended_action": action,
            "reason": reason
        }

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}


# ---------------- CSV BATCH PREDICTION ----------------

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):

    if model is None:
        return {"error": "Model not loaded"}

    try:
        df = pd.read_csv(file.file)

        risk_scores = []
        risk_levels = []

        for _, row in df.iterrows():
            row_df = pd.DataFrame([row])

            risk_score, level, _, _ = compute_prediction(row_df)

            risk_scores.append(risk_score)
            risk_levels.append(level)

        df["risk_score"] = risk_scores
        df["risk_level"] = risk_levels

        output_path = "predictions.csv"
        df.to_csv(output_path, index=False)

        return FileResponse(
            output_path,
            media_type="text/csv",
            filename="predictions.csv"
        )

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}
