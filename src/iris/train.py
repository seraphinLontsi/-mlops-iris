
from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,)



ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "iris_model.joblib"


def prepare_data():
    iris = load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("=== DATA PREPARATION ===")
    print(f"Total observations : {len(X)}")
    print(f"Training samples   : {len(X_train)}")
    print(f"Testing samples    : {len(X_test)}")

    print("\nTraining features shape:", X_train.shape)
    print("Testing features shape :", X_test.shape)

    return X_train, X_test, y_train, y_test



def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42,)
    model.fit(X_train, y_train)

    print("\n=== MODEL TRAINING ===")
    print("Random Forest trained successfully!")
    print(f"Number of trees: {model.n_estimators}")
    print(f"Number of features: {model.n_features_in_}")

    return model



def evaluate_model(model, X_test, y_test):
    # Prédictions sur les données de test
    y_pred = model.predict(X_test)

    # Calcul de l'accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\n=== MODEL EVALUATION ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy (%): {accuracy * 100:.2f}%")

    # Rapport de classification
    print("\n=== CLASSIFICATION REPORT ===")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "setosa",
                "versicolor",
                "virginica",
            ],
        )
    )

    # Matrice de confusion
    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_test, y_pred))

    return accuracy


def save_model(model, accuracy):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True,)

    artifact = {"model": model, "target_names": ["setosa", "versicolor", "virginica",],
        "feature_names": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)",],
        "accuracy": float(accuracy),
        "model_version": "1.0.0",}

    joblib.dump(artifact, MODEL_PATH)

    print("\n=== MODEL SAVING ===")
    print(f"Model saved successfully: {MODEL_PATH}")

    return MODEL_PATH



if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()

    model = train_model(X_train, y_train)

    model_eval = evaluate_model(model=model, X_test=X_test, y_test=y_test)

    accuracy = evaluate_model(model, X_test, y_test)

    save_model(model, accuracy)

