"""
============================================================
PASO 2: API de FastAPI
Archivo: app.py
============================================================

API REST que expone el modelo de clasificación Iris.

Endpoints:
    GET  /         -> Información de la API y ejemplo de uso.
    POST /predict  -> Predicción de la especie a partir de las 4 features.
    GET  /health   -> Health check para Render/Railway.
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# -------------------------------------------------------
# Inicialización
# -------------------------------------------------------
app = FastAPI(
    title="Iris Classification API",
    description="API REST que clasifica flores Iris usando un modelo RandomForest.",
    version="1.0.0",
)

# Cargar el modelo al arrancar la aplicación
model = joblib.load("model.pkl")

# Clases del dataset Iris (en el mismo orden que sklearn)
CLASSES = ["setosa", "versicolor", "virginica"]


# -------------------------------------------------------
# Esquemas Pydantic para validar las entradas y salidas
# -------------------------------------------------------
class IrisInput(BaseModel):
    """Datos de entrada para la predicción."""

    sepal_length: float = Field(..., ge=0, le=15, example=5.1)
    sepal_width: float = Field(..., ge=0, le=15, example=3.5)
    petal_length: float = Field(..., ge=0, le=15, example=1.4)
    petal_width: float = Field(..., ge=0, le=15, example=0.2)


class Probabilities(BaseModel):
    setosa: float
    versicolor: float
    virginica: float


class PredictionOutput(BaseModel):
    """Respuesta del endpoint /predict."""

    prediction: str
    prediction_index: int
    probabilities: Probabilities
    confidence: float
    status: str


# -------------------------------------------------------
# TODO 5: Endpoint raíz '/'
# -------------------------------------------------------
@app.get("/")
def home():
    """Información general de la API y ejemplo de llamada."""
    return {
        "name": "Iris Classification API",
        "version": "1.0.0",
        "algorithm": "RandomForestClassifier",
        "description": "API que predice la especie de una flor Iris a partir de 4 medidas.",
        "features_expected": [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ],
        "classes": CLASSES,
        "endpoints": {
            "/": "GET - Información de la API",
            "/predict": "POST - Predicción de la especie",
            "/health": "GET - Health check",
        },
        "example_request": {
            "url": "/predict",
            "method": "POST",
            "body": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
        },
    }


# -------------------------------------------------------
# TODO 6: Endpoint '/predict'
# -------------------------------------------------------
@app.post("/predict", response_model=PredictionOutput)
def predict(data: IrisInput):
    """
    Recibe las 4 features de Iris y devuelve la especie predicha
    junto con las probabilidades de cada clase.
    """
    try:
        # Construir el array en el orden esperado por sklearn
        features = np.array([[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width,
        ]])

        # Predicción y probabilidades
        prediction_index = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

        prediction_label = CLASSES[prediction_index]
        confidence = float(np.max(probabilities))

        return {
            "prediction": prediction_label,
            "prediction_index": prediction_index,
            "probabilities": {
                "setosa": round(float(probabilities[0]), 4),
                "versicolor": round(float(probabilities[1]), 4),
                "virginica": round(float(probabilities[2]), 4),
            },
            "confidence": round(confidence, 4),
            "status": "success",
        }

    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Falta el campo obligatorio: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno: {exc}",
        )


# -------------------------------------------------------
# TODO 7: Endpoint '/health'
# -------------------------------------------------------
@app.get("/health")
def health():
    """Health check usado por Render/Railway para verificar el servicio."""
    return {"status": "healthy"}


# -------------------------------------------------------
# Arranque local (solo si se ejecuta `python app.py`)
# En producción Render/Railway usan uvicorn directamente
# -------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    # Render/Railway asignan el puerto mediante la variable PORT
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
