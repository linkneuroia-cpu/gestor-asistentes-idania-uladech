# Imagen de producción del gestor + asistentes (Funcionalidad 1).
#
# Sin GPU confirmada en el servidor de despliegue (10.0.0.92) — se usa el
# build CPU-only de torch (el que instala requirements.txt por defecto
# desde PyPI normal; el build CUDA solo se instala si alguien corre el
# `pip install --index-url .../cu126 ...` manual descrito en
# requirements.txt, que acá NO se ejecuta). El código ya cae a CPU
# automáticamente si no detecta CUDA (ver strategies/dense_embedding.py y
# strategies/rerank.py) — más lento que con GPU, pero funciona igual.
#
# Si más adelante se confirma GPU en el servidor: cambiar la imagen base a
# una con CUDA runtime (p.ej. nvidia/cuda:12.6.0-runtime-ubuntu22.04 con
# Python instalado aparte, o una imagen python con cuda) y agregar el
# `pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.13.0`
# después del install de requirements.txt.
FROM python:3.10-slim

# Librerías de sistema que necesitan EasyOCR/pix2tex (dependen de OpenCV,
# que sin estas falla al importar en una imagen "slim") y el procesamiento
# de audio/video (faster-whisper/PyAV).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8100
CMD ["python", "-u", "app.py"]
