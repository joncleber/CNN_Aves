from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "fine_tuning_galinhas"
    / "weights"
    / "best.pt"
)

DATA_YAML = BASE_DIR / "dataset" / "data.yaml"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset não encontrado: {DATA_YAML}")

model = YOLO(str(MODEL_PATH))

metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    batch=2,
    plots=True
)

print("\n=== RESULTADOS NO CONJUNTO DE TESTE ===")
print(f"Precisão: {metrics.box.mp:.4f}")
print(f"Recall:   {metrics.box.mr:.4f}")
print(f"mAP@50:   {metrics.box.map50:.4f}")
print(f"mAP@50-95:{metrics.box.map:.4f}")
