# HeightEstimationStereoVision
python rectification.py --input_type image --input_images_folder ../images/laser/groundTruth --output_images_folder ../images/calibration_results/matlab_1 --xml ../config_files/matlab_1/newStereoMap.xml



# TODO:
## 07/25/2024 BY: Elihan
No se estan generando las nubes de puntos filtradas
## Error ouput:
Procesando situación: 200 y 400 sentados | Variante a
Found 1 images. Saving files to raft_demo_output/
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.37s/it] 
Found 1 images. Saving files to seletive_demo_output/
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.55s/it] 

0: 384x640 2 persons, 6.0ms
Speed: 2.5ms preprocess, 6.0ms inference, 2.0ms postprocess per image at shape (1, 3, 384, 640)
Results saved to runs\pose\predict3
Probabilidades: tensor([0.9363, 0.8798], device='cuda:0')
Error procesando 200 y 400 sentados: Found array with 0 sample(s) (shape=(0, 3)) while a minimum of 1 is required by DBSCAN.

# TODO:
## 07/26/2024 BY: Elihan
Ya se estan generando las nubes filtradas, te quedaste en el archivo pc_generation_ML.py en l alinea 123 justo al inicio de la funcion process_point_cloud, la solucion parecio ser cambiar eps=30 a eps=300 en generate_filtered_point_cloud.

Ahora bien, hay un error respecto a una generacion filtrada, podria ser:
A: YOLO no esta detectando a la persona (poco probable porque se bajo la probabilidad necesaria para la deteccion)
B: m_initial = 50 es un valor demasiado pequeño considerando las dimensiones de las nuevas nubes de puntos generadas.
## Error output:
Procesando situación: 250 y 350 | Variante a
Found 1 images. Saving files to raft_demo_output/
100%|███████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.12s/it] 

0: 384x640 2 persons, 6.5ms
Speed: 1.0ms preprocess, 6.5ms inference, 2.0ms postprocess per image at shape (1, 3, 384, 640)
Para el centroide con z = 1358.5508602366729, el rango de Y es: Y_min = -176.52374609411697, Y_max = 481.7331212114009
La altura de la persona 1 es de 658.2568673055179

No se encontraron puntos en el rango óptimo para este centroide.
### Error aditional observation
Parece ser que es la unica situacion en la que ocurre este error output
