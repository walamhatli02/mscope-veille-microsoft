# Image Python légère
FROM python:3.11-slim

# Dossier de travail
WORKDIR /app

# Copier les dépendances d'abord (cache Docker)
COPY requirements.txt .

# PyTorch en version CPU d'abord (~200 Mo au lieu de ~2 Go avec CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p data

# Port Streamlit
EXPOSE 8501

# Lancer l'application
CMD ["streamlit", "run", "chatbot.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
