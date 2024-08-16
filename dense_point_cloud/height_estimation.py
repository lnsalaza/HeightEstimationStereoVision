import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def calculate_scaled_positions(points, reference_point, scale_factor):
    # Restar el punto de referencia a todos los puntos
    shifted_points = points - reference_point
    
    # Escalar los puntos
    scaled_points = scale_factor * shifted_points
    
    # Volver a mover los puntos al sistema de referencia original
    new_positions = scaled_points + reference_point
    
    return new_positions

def process_chunk(chunk, reference_point, scale_factor):
    return calculate_scaled_positions(chunk, reference_point, scale_factor)

def main():
    # Genera un millón de puntos aleatorios en un espacio 3D
    num_points = 1000000
    points = np.random.rand(num_points, 3)
    
    # Define el punto de referencia
    reference_point = np.array([0, 0, 0])
    
    # Define el factor de escala deseado (por ejemplo, doblar la distancia)
    scale_factor = 2.0
    
    # Divide los puntos en trozos para el procesamiento paralelo
    num_threads = 8  # Ajusta este número según la cantidad de núcleos de tu CPU
    chunks = np.array_split(points, num_threads)
    
    new_positions = None

    # Procesa cada trozo en paralelo
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_chunk, chunk, reference_point, scale_factor) for chunk in chunks]
        
        for future in as_completed(futures):
            result = future.result()
            new_positions = np.concatenate((new_positions, result)) if new_positions is not None else result

    print("Primeros 10 puntos originales:\n", points[:10])
    print("Primeros 10 nuevos puntos escalados:\n", new_positions[:10])

if __name__ == "__main__":
    main()
