# ============================================================
# Dockerfile para la API de clasificación Iris
# ============================================================

# Imagen base ligera con Python 3.11
FROM python:3.11-slim

# Variables de entorno recomendadas para Python en contenedores
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar primero los requirements (aprovecha la caché de Docker:
# si requirements.txt no cambia, no se reinstalan dependencias)
COPY requirements.txt .

# Instalar dependencias
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar el resto del código y el modelo entrenado
COPY app.py .
COPY model.pkl .

# Render/Railway asignan el puerto en la variable de entorno PORT.
# Lo declaramos por documentación; el bind real se hace en CMD.
EXPOSE 8000

# Arranque en producción con uvicorn.
# Usamos `sh -c` para que la variable $PORT se expanda en runtime.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
