import pytest

from fastapi.testclient import TestClient
from iris.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_model(client):
    response = client.get("/model/info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_version"] == "1.0.0"
    assert 0 <= data["accuracy"] <= 1
    assert len(data["target_names"]) == 3
    assert len(data["feature_names"]) == 4


def test_model_info(client):
    response = client.get("/model/info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_version"] == "1.0.0"
    assert 0 <= data["accuracy"] <= 1
    assert len(data["target_names"]) == 3
    assert len(data["feature_names"]) == 4


def test_predict_setosa_api(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["species"] == "setosa"
    assert 0 <= data["confidence"] <= 1
    assert data["model_version"] == "1.0.0"


def test_predict_invalid_api(client):
    payload = {
        "sepal_length": -5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422





