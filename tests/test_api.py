from fastapi.testclient import TestClient
import pandas as pd
from api.main import app

client = TestClient(app)

def test_predict_endpoint_minimal():
    data = [
        {"datetime": "2025-01-01T00:00:00", "price": 100.0},
        {"datetime": "2025-01-01T00:01:00", "price": 101.0},
        {"datetime": "2025-01-01T00:02:00", "price": 100.5},
        {"datetime": "2025-01-01T00:03:00", "price": 101.5},
    ]
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    body = response.json()
    # 各キーが含まれる
    assert set(body.keys()) == {"datetime", "trend", "vol", "signal"}
    # シグナルリストの長さは入力長-1
    assert len(body["signal"]) == len(data) - 1
