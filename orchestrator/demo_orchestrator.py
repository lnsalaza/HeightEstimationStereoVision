import cv2
from orchestrator import Orchestrator

def demo_orchestrator():
    # Cargar las imágenes iniciales
    img_left_array = cv2.imread("../originals/Steven_depth/300/14_24_25_19_08_2024_IMG_LEFT.jpg")
    img_right_array = cv2.imread("../originals/Steven_depth/300/14_24_25_19_08_2024_IMG_RIGHT.jpg")
    

    # Inicializar el Orchestrator con las imágenes cargadas y parámetros iniciales
    orchestrator = Orchestrator(img_left_array, img_right_array, initial_requirement="dense", profile_name="default_profile", method="SGBM", normalize=True, use_max_disparity=True)
    
    while True:
        # Solicitar el nuevo requerimiento
        new_requirement = input("\nIntroduce el nuevo requerimiento (dense/nodense/features/height) o 'exit' para salir: ").strip().lower()
        if new_requirement == "exit":
            print("Saliendo del orquestador...")
            break

        orchestrator.set_requirement(new_requirement)

        # Preguntar si se quiere cambiar las imágenes
        change_images = input("¿Quieres cambiar las imágenes? (y/n): ").strip().lower()
        if change_images == 'y':
            img_left_path = input("Introduce la ruta de la nueva imagen izquierda: ")
            img_right_path = input("Introduce la ruta de la nueva imagen derecha: ")
            img_left_array_new = cv2.imread(img_left_path)
            img_right_array_new = cv2.imread(img_right_path)
            orchestrator.set_images(img_left_array_new, img_right_array_new)

        # Preguntar si se quiere cambiar el perfil
        change_profile = input("¿Quieres cambiar el perfil de calibración? (y/n): ").strip().lower()
        if change_profile == 'y':
            profile_name = input("Introduce el nombre del nuevo perfil: ").strip()
            orchestrator.set_profile(profile_name)

        # Preguntar si se quiere cambiar el método de disparidad, normalización y disparidad máxima
        change_method_params = input("¿Quieres cambiar el método de disparidad o parámetros? (y/n): ").strip().lower()
        if change_method_params == 'y':
            method = input("Introduce el método de disparidad ('SGBM', 'RAFT', 'SELECTIVE'): ").strip()
            normalize = input("¿Quieres normalizar la nube de puntos? (True/False): ").strip().lower() == "true"
            use_max_disparity = input("¿Quieres usar la disparidad máxima? (True/False): ").strip().lower() == "true"
            orchestrator.set_method_params(method, normalize, use_max_disparity)

        try:
            # Ejecutar el módulo basado en el requerimiento actual
            result = orchestrator.execute()
            print("Resultado:", result)
        except FileNotFoundError as e:
            print(f"Error: {str(e)}")
        except Exception as e:
            print(f"Ocurrió un error: {str(e)}")

if __name__ == "__main__":
    demo_orchestrator()
