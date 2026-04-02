import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient

os.environ["TESTING"] = "true"

from serve import app
client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200

def test_root():
    res = client.get("/")
    assert res.status_code == 200

def test_predict():
    res = client.post("/predict", json={
        "features": [13.2, 2.77, 2.51, 18.5, 96.6, 1.04,
                     2.55, 0.57, 1.47, 6.2, 1.05, 3.33, 820.0]
    })
    assert res.status_code == 200
    data = res.json()
    assert "prediction" in data
    assert "probabilities" in data
    assert len(data["probabilities"]) == 3

def test_predict_wrong_features():
    res = client.post("/predict", json={"features": [1.0, 2.0]})
    assert res.status_code == 422
