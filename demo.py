import cv2
import torch
import time
import threading
from ultralytics import YOLO
from orchestrator.orchestrator import Orchestrator

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def detect_person_with_orchestrator_webcam():
    # Inicializar la cámara
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

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

            # Mostrar el frame original
            cv2.imshow('Webcam Feed', frame)

            # Detectar personas en el frame (esto es opcional y depende de tu pipeline de detección)
            result = detect_person_in_frame(frame)

            if result:  # Si se detecta una persona
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


def detect_person_in_frame(frame):
    # Simular una detección de persona
    return True


if __name__ == "__main__":
    detect_person_with_orchestrator_webcam()
