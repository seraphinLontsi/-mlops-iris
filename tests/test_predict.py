from iris.predict import load_model, predict_flower
import pytest


def test_predict_setosa():

    artifact = load_model()
    features = [5.1, 3.5, 1.4, 0.2]
    result = predict_flower(features=features, artifact=artifact)

    assert result['species'] == "setosa"
    assert 0 <= result['confidence'] <= 1
    assert result['model_version'] == "1.0.0"


def test_predict_invalid_features():
    features = [5.1, 3.5, 1.4]
    with pytest.raises(ValueError):
        predict_flower(features=features)