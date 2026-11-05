
from pathlib import Path
from ultralytics import YOLO

# Diretório do projeto
BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "dataset" / "data.yaml"

print("Pasta do projeto:", BASE_DIR)
print("Arquivo YAML:", DATA_YAML)
print("YAML existe?", DATA_YAML.exists())

# 1. Carrega o modelo pré-treinado
model = YOLO("../yolov8n.pt")

#2. Executa o fine-tuning
results = model.train(
    data=str(DATA_YAML),
    epochs=80,
    imgsz=640,
    batch=4,
    patience=15,
    workers=2,
    seed=42,
    deterministic=True,
    project=str(BASE_DIR / "runs"),
    name="fine_tuning_galinhas",
    exist_ok=True,
    plots=True,
    save=True,
    pretrained=True
)

print("Treinamento concluído!")
print("Verifique a pasta runs/detect/fine_tuning_galinhas")
