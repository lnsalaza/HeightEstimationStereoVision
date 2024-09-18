import cv2
import torch
import time
import threading
from ultralytics import YOLO
from orchestrator.orchestrator import Orchestrator

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

altura = 0
class HeightEstimatorThread(threading.Thread):
    def __init__(self, orchestrator, img_left, img_right):
        threading.Thread.__init__(self)
        self.orchestrator = orchestrator
        self.img_left = img_left
        self.img_right = img_right

    def run(self):
        self.orchestrator.set_images(self.img_left, self.img_right)
        print("Procesando altura en segundo plano...")
        result = self.orchestrator.execute()
        print(f"Resultado de altura: {result}")
        altura = result[0]['height']
        print(altura)

def detect_person_with_orchestrator_webcam():
    # Inicializar la cámara
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    # Inicializar el modelo YOLOv8
    model = YOLO('yolov8n.pt').to(device)

    orchestrator = Orchestrator(img_left=None, img_right=None, profile_name="MATLAB", method="SELECTIVE", normalize=True, use_max_disparity=True, requirement="height")
    last_processed_time = 0  # Para evitar procesar en cada frame
    processing_interval = 10  # Segundos
    height_thread = None  # Hilo para la estimación de altura

    try:
        while True:
            # Capturar el frame
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo capturar el frame.")
                break

            # Realizar la predicción con YOLO
            results = model(frame, conf=0.8)

            # Dibujar las cajas de detección y probabilidades
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas de la caja
                    conf = box.conf[0]  # Confianza de detección

                    # Dibujar la caja
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {conf:.2f} - Height {altura}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Mostrar el frame con las detecciones
            cv2.imshow('YOLOv8 Person Detection', frame)

            # Si se detecta una persona
            if len(results[0].boxes) > 0:
                current_time = time.time()

                # Procesar cada 'processing_interval' segundos y solo si no hay otro hilo en ejecución
                if current_time - last_processed_time > processing_interval and (height_thread is None or not height_thread.is_alive()):
                    print("Iniciando procesamiento de altura...")

                    # Capturar y convertir las imágenes a NumPy arrays
                    ret, img_left = cap.read()  # Simular la imagen izquierda con el frame actual
                    ret, img_right = cap.read()  # Simular la imagen derecha con el frame siguiente

                    # Asegurarse de que las imágenes están en formato NumPy array
                    if img_left is not None and img_right is not None:
                        # Crear un nuevo hilo para la estimación de altura
                        height_thread = HeightEstimatorThread(orchestrator, img_left, img_right)
                        height_thread.start()

                    # Actualizar el tiempo de la última ejecución
                    last_processed_time = current_time

            # Salir del bucle si se presiona 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Detenido por el usuario.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_person_with_orchestrator_webcam()
