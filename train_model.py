"""
============================================================
PASO 1: Entrenar y guardar el modelo
Archivo: train_model.py
============================================================

Este script entrena un clasificador RandomForest sobre el
dataset Iris y guarda el modelo entrenado en 'model.pkl'.
"""

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# -------------------------------------------------------
# Cargar el dataset Iris
# -------------------------------------------------------
iris = load_iris()
X, y = iris.data, iris.target

print(f"Dataset Iris cargado: {X.shape[0]} muestras, {X.shape[1]} features")
print(f"Clases: {list(iris.target_names)}")
print()

# -------------------------------------------------------
# TODO 1: Divide los datos en entrenamiento y prueba
# -------------------------------------------------------
# Usamos 80% para entrenamiento y 20% para test.
# stratify=y asegura que las 3 clases estén balanceadas en ambos sets.
# random_state=42 garantiza reproducibilidad.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train set: {X_train.shape[0]} muestras")
print(f"Test set:  {X_test.shape[0]} muestras")
print()

# -------------------------------------------------------
# TODO 2: Elige y entrena un algoritmo de clasificación
# -------------------------------------------------------
# Elegimos RandomForestClassifier porque:
#  - Maneja muy bien datasets pequeños como Iris (150 muestras)
#  - Robusto ante outliers y ruido
#  - No requiere escalado de features
#  - Devuelve probabilidades (predict_proba) que necesitamos en la API
#  - Suele alcanzar accuracy >95% en Iris sin tuning agresivo
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)
print("Modelo entrenado correctamente.")
print()

# -------------------------------------------------------
# TODO 3: Evalúa el modelo y muestra métricas
# -------------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy en test: {accuracy:.4f}")
print()
print("Classification report:")
print(classification_report(
    y_test, y_pred,
    target_names=iris.target_names
))

# -------------------------------------------------------
# TODO 4: Guarda el modelo como 'model.pkl'
# -------------------------------------------------------
joblib.dump(model, 'model.pkl')

print('¡Modelo guardado correctamente como model.pkl!')
