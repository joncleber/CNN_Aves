from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

model_path = (
    BASE_DIR
    / "runsV2"
    / "fine_tuning_galinhasV2"
    / "weights"
    / "best.pt"
)

#image_path = BASE_DIR / "dataset" / "test" / "images"
image_path = BASE_DIR / "Imagens_avulsa"

# Pasta onde as imagens analisadas serão salvas
output_dir = BASE_DIR / "resultadosV2"

model = YOLO(str(model_path))

results = model.predict(
    source=str(image_path),
    imgsz=640,
    conf=0.05,
    save=True,
    project=str(output_dir),
    name="galinheiro",
    exist_ok=True
)

print("Imagens processadasV2:", len(results))
print("Resultados salvos em:", results[0].save_dir if results else "Nenhum")
