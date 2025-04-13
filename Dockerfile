FROM python:3.9

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Création d'un répertoire temporaire accessible
RUN mkdir -p /tmp/pip-cache && chmod 777 /tmp/pip-cache

# Configuration de pip pour utiliser un cache personnalisé
ENV PIP_CACHE_DIR=/tmp/pip-cache
ENV PIP_NO_CACHE_DIR=off

# Copie des requirements
COPY requirements.txt .

# Installation des dépendances Python
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie du reste de l'application
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]