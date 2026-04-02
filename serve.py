import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os

app = FastAPI(title="Wine Classifier API", version="1.0")

TESTING = os.environ.get("TESTING") == "true"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_export")

if TESTING:
    model = None
    print("🧪 테스트 모드: 모델 로드 생략")
else:
    try:
        model = mlflow.sklearn.load_model(MODEL_PATH)
        print("✅ 모델 로드 성공")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        model = None

class PredictRequest(BaseModel):
    features: List[float]

class PredictResponse(BaseModel):
    prediction: int
    wine_type: str
    probabilities: List[float]

WINE_LABELS = {0: "Class 0 (Barolo)", 1: "Class 1 (Grignolino)", 2: "Class 2 (Barbera)"}

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.features) != 13:
        raise HTTPException(status_code=422, detail=f"피처 13개 필요, {len(req.features)}개 입력됨")

    if TESTING:
        return PredictResponse(
            prediction=0,
            wine_type=WINE_LABELS[0],
            probabilities=[0.98, 0.01, 0.01]
        )

    if model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    X = np.array(req.features).reshape(1, -1)
    pred  = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0].tolist()
    return PredictResponse(
        prediction=pred,
        wine_type=WINE_LABELS[pred],
        probabilities=[round(p, 4) for p in proba]
    )
