from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

model_path = (
    BASE_DIR
    / "runs"
    / "fine_tuning_galinhas"
    / "weights"
    / "best.pt"
)

# Coloque uma imagem de teste nesta pasta.
#image_path = BASE_DIR / "dataset" / "test" / "images"
image_path = BASE_DIR / "Imagens_avulsa"

# Pasta onde as imagens analisadas serão salvas
output_dir = BASE_DIR / "resultados"

model = YOLO(str(model_path))

results = model.predict(
    source=str(image_path),
    imgsz=640,
    conf=0.25,
    save=True,
    project=str(output_dir),
    name="galinheiro",
    exist_ok=True
)

print("Imagens processadas:", len(results))
print("Resultados salvos em:", results[0].save_dir if results else "Nenhum")
