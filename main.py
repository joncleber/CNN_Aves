import cv2
import numpy as np
from ultralytics import YOLO

# 1. Carregar modelo YOLO treinado na suas classes de aves
# Treine o YOLOv8 com fotos anotadas do seu próprio galinheiro
model = YOLO('yolov8n.pt')

# Fator de conversão fictício (cm por pixel) - Calibrar no local real
# Exemplo: 1 pixel representa 0.15 cm à distância média da câmera
PIXEL_TO_CM = 0.015

def estimar_peso(area_cm2, classe):
    """
    Função de regressão simples para estimar o peso baseado na área.
    Ajuste os coeficientes (a, b) com base na amostragem real da sua granja.
    """
    coeficientes = {
        'pintinho': (0.05, 1.1),
        'galinha': (0.12, 1.2),
        'galo': (0.15, 1.25),
        'pato': (0.14, 1.2)
    }
    a, b = coeficientes.get(classe, (0.1, 1.2))
    peso_kg = a * (area_cm2 ** (b / 2)) / 100
    return max(0.05, round(peso_kg, 2))

# Open video feed (arquivo ou RTSP da câmera)
cap = cv2.VideoCapture('1788822649578_0.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar inferência no frame
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Coordenadas e identificação
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label_class = model.names[cls_id]
            conf = float(box.conf[0])

            if conf < 0.5:
                continue

            # 1. Estimativa de Tamanho (Largura e Altura em cm)
            largura_px = x2 - x1
            altura_px = y2 - y1
            largura_cm = largura_px * PIXEL_TO_CM
            altura_cm = altura_px * PIXEL_TO_CM

            # 2. Estimativa de Área e Peso
            area_cm2 = largura_cm * altura_cm
            peso_est = estimar_peso(area_cm2, label_class)

            # 3. Status de Saúde Simulado (Ex: se altura/largura muito desproporcional ou parado)
            status_saude = "Saudavel"  # Pode integrar com tracking ByteTRACK para medir mobilidade

            # Desenhar Bounding Box e Informações no Frame
            color = (0, 255, 0) if status_saude == "Saudavel" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            info_text = f"{label_class} | {altura_cm:.1f}cm | ~{peso_est}kg"
            cv2.putText(frame, info_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Monitoramento Granja - Visao Computacional', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()