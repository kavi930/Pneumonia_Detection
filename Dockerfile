
# ============================================================
# CREATE DOCKERFILE
# ============================================================

dockerfile = """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY best_model.keras .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
"""

DOCKER_PATH = os.path.join(
    DEPLOY_DIR,
    "Dockerfile"
)

with open(DOCKER_PATH, "w") as f:
    f.write(dockerfile)

print("Dockerfile created")
