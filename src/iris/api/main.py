
from contextlib import asynccontextmanager
from iris.predict import load_model
from fastapi import FastAPI, Request
from iris.api.schemas import IrisFeatures
from iris.predict import predict_flower



@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Iris ML model...")

    artifact = load_model()

    app.state.model_artifact = artifact

    print("Iris ML model loaded successfully!")
    yield
    print("Shutting down Iris API...")


app = FastAPI(
    title="Iris Classification API",
    description="Machine Learning API for Iris classification",
    version="1.0.0",
    lifespan=lifespan,)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/model/info")
def model_info(request: Request):
    artifact = request.app.state.model_artifact

    return {
        "model_version": artifact["model_version"],
        "accuracy": artifact["accuracy"],
        "target_names": artifact["target_names"],
        "feature_names": artifact["feature_names"],
    }

@app.get("/")
def root():
    return {
        "message": "Welcome to Iris Classification API",
        "version": "1.0.0",
        "documentation": "/docs",
    }

@app.post("/predict")
def predict(features: IrisFeatures, request: Request):
    
    artifact = request.app.state.model_artifact

    mesurements = [features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]

    result = predict_flower(mesurements, artifact=artifact)

    return result
