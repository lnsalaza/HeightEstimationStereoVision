import cv2
from ultralytics import YOLO
import torch

torch.cuda.set_device(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def detect_person_webcam(model_path: str = 'yolov8n.pt', conf_threshold: float = 0.8):
    # Cargar el modelo YOLOv8
    model = YOLO(model_path).to(device=device)

    # Inicializar la cámara web
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    try:
        while True:
            # Capturar frame de la cámara
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer el frame.")
                break

            # Realizar la predicción con el umbral de confianza configurado
            results = model(frame, conf=conf_threshold)

            # Dibujar las cajas alrededor de las personas detectadas
            for result in results:
                for box in result.boxes:
                    # Obtener las coordenadas de la caja y la confianza
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()

                    # Dibujar la caja en la imagen
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Mostrar el frame con las detecciones
            cv2.imshow('YOLOv8 Person Detection', frame)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrumpido por el usuario.")

    finally:
        # Liberar los recursos de la cámara
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_person_webcam()