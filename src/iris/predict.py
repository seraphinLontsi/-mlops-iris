from pathlib import Path
import joblib
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "iris_model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileExistsError(f"Model not found: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)
    return artifact


def predict_flower(features, artifact=None):

    values = np.array(features, dtype=float)

    if values.shape != (4,):
        raise ValueError("Exactly four measurements are required.")

    if not np.all(np.isfinite(values)):
        raise ValueError("All measurements must be finite numbers.")

    if np.any(values <= 0):
        raise ValueError( "All measurements must be positive.")

    if artifact is None:
        artifact = load_model()

    model = artifact["model"]
    target_names = artifact["target_names"]

    X = values.reshape(1,-1)

    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]

    return {"species": target_names[prediction], "confidence": round(float(probabilities[prediction]), 4), "model_version": artifact["model_version"],}



if __name__ == "__main__":
    sample = [5.1, 3.5, 1.4, 0.2]

    result = predict_flower(sample)

    print("\n=== IRIS PREDICTION ===")
    print(f"Species: {result['species']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Model version: {result['model_version']}")