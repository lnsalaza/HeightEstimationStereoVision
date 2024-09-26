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
            # Se rota la captura del video 180 grados
            frame = cv2.rotate(frame, cv2.ROTATE_180)

            # Se corta la resolucion de 3840 a la mitad para que queda 1080
            middle_width = frame.shape[1] // 2

            # Se obtienen las 2 capturas de la camara, teniendo en cuenta que el posicionamiento esta invertido
            frame_l = frame[:, middle_width:]
            frame_r = frame[:, :middle_width]
            
            # Realizar la predicción con YOLO
            results = model(frame_l, conf=0.8)[0]

            # Dibujar las cajas de detección y probabilidades
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas de la caja
                    conf = box.conf[0]  # Confianza de detección

                    # Dibujar la caja
                    cv2.rectangle(frame_l, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_l, f"Height {altura} - Person {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            # Si se detecta una persona
                

                # Salir del bucle si se presiona 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Mostrar el frame con las detecciones
            cv2.imshow('YOLOv8 Person Detection', frame_l)
            if len(results.boxes) > 0:
                    current_time = time.time()

                    # Procesar cada 'processing_interval' segundos y solo si no hay otro hilo en ejecución
                    if current_time - last_processed_time > processing_interval and (height_thread is None or not height_thread.is_alive()):
                        print("Iniciando procesamiento de altura...")

                        
                        # Asegurarse de que las imágenes están en formato NumPy array
                        if frame_l is not None and frame_r is not None:
                            # Crear un nuevo hilo para la estimación de altura
                            height_thread = HeightEstimatorThread(orchestrator, frame_l, frame_r)
                            try:
                                cv2.imshow(frame_l, 0)
                            except:
                                print("error en la imagen")
                            height_thread.start()

                        # Actualizar el tiempo de la última ejecución
                        last_processed_time = current_time

    except KeyboardInterrupt:
        print("Detenido por el usuario.")
        print(altura)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_person_with_orchestrator_webcam()
