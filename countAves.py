"""
countAves.py — Contagem de aves em imagem estática com YOLOv8 V1.

Edite os caminhos e as opções na seção CONFIGURAÇÃO.
"""

from pathlib import Path
from collections import Counter
import csv
import re
import unicodedata

import cv2
from ultralytics import YOLO

# ============================================================
# 1. CONFIGURAÇÃO — AJUSTE ESTES CAMINHOS
# ============================================================

MODEL_PATH = Path(
    "/Users/cleber/PycharmProjects/Projeto_Integrador/runs/fine_tuning_galinhas/weights/best.pt"
)

IMAGE_PATH = Path(
    "/Users/cleber/PycharmProjects/Projeto_Integrador/Imagens_avulsa/imagem.jpeg"
)

OUTPUT_DIR = Path(
    "/Users/cleber/PycharmProjects/Projeto_Integrador/resultados_contagem"
)

IMAGE_SIZE = 640
IOU_THRESHOLD = 0.45

# Classes do V1 que entram na contagem.
BIRD_CLASSES = {"frango", "galinha", "galo", "pato", "pintinho"}

# ============================================================
# 2. MODO DE EXECUÇÃO — ATIVE SOMENTE O QUE DESEJAR
# ============================================================

# Comparação automática: conf=0.10, 0.25 e 0.40.
RUN_COMPARISON = True

# Execução individual: para usar, desative RUN_COMPARISON.
RUN_SINGLE = False
SINGLE_CONFIDENCE = 0.25

# False = somente classes de aves; True = todas as classes.
COUNT_ALL_CLASSES = False

CONFIDENCE_LEVELS = [0.10, 0.25, 0.40]


# ============================================================
# 3. FUNÇÕES AUXILIARES
# ============================================================

def normalize_name(name):
    name = str(name).strip().lower()
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"[_-]+", " ", name)
    return " ".join(name.split())


def get_class_name(names, class_id):
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def find_bird_class_ids(model):
    configured = {normalize_name(name) for name in BIRD_CLASSES}
    return [
        class_id
        for class_id in range(len(model.names))
        if normalize_name(get_class_name(model.names, class_id)) in configured
    ]


# ============================================================
# 4. DETECÇÃO E CONTAGEM
# ============================================================

def process_image(model, image, confidence, bird_class_ids):
    print("\n" + "-" * 55)
    print(f"Processando com conf={confidence:.2f}")

    selected_classes = None if COUNT_ALL_CLASSES else bird_class_ids

    result = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=confidence,
        iou=IOU_THRESHOLD,
        classes=selected_classes,
        verbose=False,
    )[0]

    counts = Counter()
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            score = float(box.conf.item())
            name = get_class_name(model.names, class_id)
            counts[name] += 1
            coords = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            detections.append((name, score, coords))

    total = sum(counts.values())
    print("Contagem por classe:")
    for name, quantity in sorted(counts.items()):
        print(f"  {name}: {quantity}")
    if not counts:
        print("  Nenhuma ave detectada.")
    print(f"TOTAL DE AVES DETECTADAS: {total}")

    annotated = result.plot()
    output_path = OUTPUT_DIR / f"{IMAGE_PATH.stem}_conf_{confidence:.2f}.jpg"
    if cv2.imwrite(str(output_path), annotated):
        print(f"Imagem anotada salva: {output_path}")
    else:
        print(f"ERRO ao salvar imagem: {output_path}")

    print("Detecções individuais:")
    for index, (name, score, coords) in enumerate(detections, 1):
        print(f"  {index:02d}. {name} | confiança={score:.3f} | caixa={coords}")

    return {
        "imagem": IMAGE_PATH.name,
        "confianca_minima": confidence,
        "total_aves_detectadas": total,
        "contagem_por_classe": "; ".join(
            f"{name}: {quantity}" for name, quantity in sorted(counts.items())
        ),
        "arquivo_resultado": str(output_path),
    }


# ============================================================
# 5. EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    print("=" * 55)
    print("CONTAGEM DE AVES — MODELO V1")
    print("=" * 55)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Modelo não encontrado: {MODEL_PATH}\n"
            "Corrija MODEL_PATH na seção 1."
        )
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f"Imagem não encontrada: {IMAGE_PATH}\n"
            "Corrija IMAGE_PATH na seção 1."
        )

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise ValueError(f"Não foi possível abrir a imagem: {IMAGE_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nCarregando modelo V1...")
    model = YOLO(str(MODEL_PATH))
    print("Classes do modelo:", model.names)

    bird_class_ids = find_bird_class_ids(model)
    if not COUNT_ALL_CLASSES and not bird_class_ids:
        raise ValueError(
            "Nenhuma classe de BIRD_CLASSES foi encontrada. "
            "Confira model.names e ajuste BIRD_CLASSES."
        )

    if COUNT_ALL_CLASSES:
        print("Contabilizando todas as classes.")
    else:
        print("Classes de aves:")
        for class_id in bird_class_ids:
            print(f"  {class_id}: {get_class_name(model.names, class_id)}")

    report = []

    # MODO A — COMPARAÇÃO AUTOMÁTICA
    if RUN_COMPARISON:
        print(f"\nComparação automática: {CONFIDENCE_LEVELS}")
        for confidence in CONFIDENCE_LEVELS:
            report.append(
                process_image(model, image, confidence, bird_class_ids)
            )

    # MODO B — UM ÚNICO LIMIAR
    if RUN_SINGLE:
        report.append(
            process_image(model, image, SINGLE_CONFIDENCE, bird_class_ids)
        )

    if not RUN_COMPARISON and not RUN_SINGLE:
        print("Nenhum modo ativo. Ative RUN_COMPARISON ou RUN_SINGLE.")
        return

    report_path = OUTPUT_DIR / "relatorio_contagem.csv"
    with report_path.open("w", newline="", encoding="utf-8-sig") as file:
        fields = [
            "imagem",
            "confianca_minima",
            "total_aves_detectadas",
            "contagem_por_classe",
            "arquivo_resultado",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report)

    print("\nPROCESSAMENTO CONCLUÍDO")
    print("Pasta de resultados:", OUTPUT_DIR.resolve())
    print("Relatório CSV:", report_path.resolve())
    print(
        "Atenção: o total é o número de caixas detectadas. "
        "Confira falsos positivos, aves sobrepostas e aves não detectadas."
    )


if __name__ == "__main__":
    main()
