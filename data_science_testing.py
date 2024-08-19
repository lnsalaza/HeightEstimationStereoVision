import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path

def read_dataset(filename):
    # Leer el archivo CSV en un DataFrame
    df = pd.read_csv(filename)
    
    # Imprimir las columnas del DataFrame
    print("Columnas del DataFrame:")
    print(df.columns.tolist())

    # Opcional: imprimir las primeras filas para ver una muestra de los datos
    print("\nFilas del DataFrame:")
    print(df)

    return df

def train_linear_regression(X, y):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear un modelo de regresión lineal
    model = LinearRegression()
    
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    # Predecir el conjunto de prueba
    y_pred = model.predict(X_test)
    
    # Calcular el error cuadrático medio y el coeficiente de determinación R^2
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Imprimir la ecuación de la regresión lineal
    print(f"y = {model.coef_[0]}*x + {model.intercept_}")
    
    return model


def extract_true_depth(row):
    # Asegurarse de que 'situation' es una cadena de texto
    situation = str(row['situation'])
    
    # Extraer solo los números de la cadena situation, considerando situaciones complejas
    numbers = re.findall(r'\d+', situation)
    if len(numbers) > 1:
        # Si hay más de un número, asumir que corresponden a diferentes personas
        return int(numbers[row['person_index'] - 1])
    elif numbers:
        # Si solo hay un número, usarlo para todas las personas
        return int(numbers[0])
    else:
        # Si no hay números, esto debería ser manejado adecuadamente; podría devolver None o un valor predeterminado
        return None


def plot_depth_time_series(filename):
    # Leer el dataset
    df = pd.read_csv(filename)
    
    # Asegurarse de que no hay valores NaN en las columnas que vamos a usar
    df = df.dropna(subset=['height', 'true_height'])

    # Obtener la lista de personas únicas
    unique_persons = df['person_index'].unique()
    
    # Colores menos saturados para depth y true_depth
    colors = ['#2467ad', '#ad3224']

    # Determinar el número de filas necesarias para los subplots
    num_persons = len(unique_persons)
    if num_persons == 1:
        num_rows, num_cols = 1, 1  # Solo una persona, un subplot
    else:
        num_cols = 2  # Dos columnas de subplots
        num_rows = (num_persons + 1) // num_cols  # Calcular el número de filas necesarias
    
    # Crear una figura grande para contener todos los subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 5 * num_rows), squeeze=False)
    axs = axs.flatten()  # Aplanar el array de axes para facilitar el manejo
    
    # Graficar una línea por persona para 'depth' y 'true_depth' en subplots individuales
    for index, person in enumerate(unique_persons):
        color = colors[1] 
        color2 = colors[0] 
        
        # Filtrar los datos para la persona actual y ordenar por 'true_depth'
        person_data = df[df['person_index'] == person].sort_values(by='true_height').reset_index(drop=True)
        
        ax = axs[index]
        ax.plot(person_data.index, person_data['height'], marker='o', linestyle='-', color=color, label=f'Person {person} [Estimated]')
        ax.plot(person_data.index, person_data['true_height'], marker='x', linestyle='--', color=color2, label=f'Person {person} [True]')
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Height')
        
        ax.legend()
        ax.grid(True)
    
    # Ajustar subplots no utilizados si hay menos personas que subplots creados
    for ax in axs[num_persons:]:
        ax.set_visible(False)  # Ocultar los subplots no utilizados

    # Añadir un título general a la figura
    fig.suptitle('WLS-SGBM Height Estimation [CORRECTED]', fontsize=16, y=0.95)
    
    plt.tight_layout()

    # Mostrar la gráfica
    plt.show()

def plot_precision_by_method(df):
    # Crear una figura con subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5), sharey=True)

    # Métodos a plotear
    methods = ['WLS-SGBM', 'RAFT-STEREO', 'SELECTIVE-IGEV']
    colors = ['#2467ad', '#ad3224', '#24ad4b']

    # Iterar sobre cada método y su correspondiente axis
    for ax, method, color in zip(axes, methods, colors):
        # Filtrar los datos para el método actual
        method_data = df[df['method'] == method]

        # Ordenar los datos por 'depth' para asegurar un gráfico correcto
        method_data = method_data.sort_values(by='situation_depth')

        # Usar fill_between para rellenar el área bajo la curva
        ax.fill_between(method_data['situation_depth'], method_data['height_precision'], color=color, alpha=0.6)
        ax.plot(method_data['situation_depth'], method_data['height_precision'], color=color, linestyle=':')
        ax.set_title(f"{method}")
        ax.set_xlabel('Depth')
        ax.set_ylabel('Height Precision (%)')
        ax.grid(True)

    fig.suptitle('Impact of Depth on Height Estimation Accuracy', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.show()

def csv_to_excel(csv_file, excel_file):
    # Leer el archivo CSV
    df = pd.read_csv(csv_file)
    
    # Guardar el DataFrame como archivo Excel
    df.to_excel(excel_file, index=False, engine='openpyxl')

    print(f"Archivo convertido con éxito de {csv_file} a {excel_file}")


def convert_csv_to_excel(input_directory, output_directory):
    # Crear un objeto Path para los directorios
    input_dir = Path(input_directory)
    output_dir = Path(output_directory)
    
    # Asegurarse de que el directorio de salida existe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Recorrer todos los archivos CSV en el directorio de entrada y subdirectorios
    for csv_file in input_dir.rglob('*.csv'):
        # Leer el archivo CSV
        df = pd.read_csv(csv_file)
        
        # Crear la ruta del archivo Excel correspondiente en el directorio de salida
        relative_path = csv_file.relative_to(input_dir)  # Obtener la ruta relativa
        excel_file = output_dir / relative_path.with_suffix('.xlsx')
        
        # Crear subdirectorios en el directorio de salida si no existen
        excel_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar el DataFrame como archivo Excel
        df.to_excel(excel_file, index=False, engine='openpyxl')
        
        print(f"Convertido {csv_file} a {excel_file}")



def join_datasets(files, labels):
    # Lista para almacenar los dataframes
    dataframes = []

    # Cargar cada dataset y añadir la columna 'Source'
    for file, label in zip(files, labels):
        df = pd.read_csv(file)
        df['method'] = label  # Añadir la columna con el nombre del dataset
        dataframes.append(df)

    # Concatenar todos los dataframes en uno solo
    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df

def calculate_height_precision(df):
    # Asegurar que 'situation_height' sea un valor numérico
    # Si 'situation_height' ya es numérico, puedes comentar la siguiente línea
    df['situation_height'] = pd.to_numeric(df['situation_height'], errors='coerce')

    # Calcular la precisión de la altura
    # La fórmula de precisión es: ((Valor real - Valor estimado) / Valor real) * 100
    # Aquí calculamos el error absoluto porcentual como la diferencia porcentual entre el ground truth y la estimación
    df['height_precision'] = 100 - (100 * abs(df['situation_height'] - df['height']) / df['situation_height'])

    return df

def calcular_promedio_height_precision(df, min_depth, max_depth, specific_method):
    """
    Calcula el promedio de height_precision para un rango de situation_depth
    y un método específico.

    Parámetros:
    df (pd.DataFrame): El DataFrame que contiene los datos.
    min_depth (float): El valor mínimo del rango de situation_depth.
    max_depth (float): El valor máximo del rango de situation_depth.
    specific_method (str): El método específico para filtrar los datos.

    Retorna:
    float: El promedio de height_precision para los filtros especificados.
    """
    # Filtrar el DataFrame por el rango de situation_depth y el método específico
    filtered_df = df[(df['situation_depth'] >= min_depth) & 
                     (df['situation_depth'] <= max_depth) & 
                     (df['method'] == specific_method)]
    
    # Calcular el promedio de height_precision
    avg_height_precision = filtered_df['height_precision'].mean()
    
    return avg_height_precision


if __name__ == "__main__":
    method = "WLS-SGBM"
    # #Suponiendo que tu archivo CSV se llama 'data.csv' y está en el mismo directorio que este script
    # filename = f"./datasets/data/estable/DEPTH/NORM-{method}_DEPTH.csv"
    # dataset = read_dataset(filename)
    # # Aplicar la función para crear la nueva columna
    # dataset['true_depth'] = dataset.apply(extract_true_depth, axis=1)

    # print(dataset)
    # dataset.to_csv(f'./datasets/data/estable/CLEANED-DEPTH/CLEANED-NORM-{method}_DEPTH.csv', index=False)
    # print("DONE")



    # # MODELO REGRESION LINEAL
    # # Cargar el dataset
    # df = pd.read_csv('./datasets/data/estable/CLEANED-WLS-SGBM_FLEXOMETRO.csv')
    
    # # Asegurarse de que no hay valores NaN en las columnas que vamos a usar
    # df = df.dropna(subset=['depth', 'true_depth'])
    
    # # Preparar los datos para la regresión
    # X = df[['depth']]  # Características (en este caso, solo una)
    # y = df['true_depth']  # Variable objetivo

    # # Entrenar el modelo
    # model = train_linear_regression(X, y)




    # GRAFICOS
    # filename = f"./datasets/data/estable/CLEANED-HEIGHT/CLEANED-NORM-{method}_HEIGHT.csv"
    # # filename_excel = './datasets/data/estable/EXCEL/NO-NORM-CLEANED-WLS-SGBM_FLEXOMETRO.xlsx'
    
    # plot_depth_time_series(filename)
    # csv_to_excel(filename, filename_excel)
    # convert_csv_to_excel("./datasets/data/estable", "./datasets/data/estable_excel")






    # # GRAFICOS EXTRA 
    # files = ['./datasets/data/estable/EXTRA/NORM-WLS-SGBM_EXTRA.csv', './datasets/data/estable/EXTRA/NORM-RAFT_EXTRA.csv', './datasets/data/estable/EXTRA/NORM-SELECTIVE_EXTRA.csv']
    # labels = ['WLS-SGBM', 'RAFT-STEREO', 'SELECTIVE-IGEV']
    # # Unir los datasets
    # result_df = join_datasets(files, labels)
    # result_df = calculate_height_precision(result_df)
    # result_df.to_csv('./datasets/data/estable/EXTRA/EXTRA.csv', index=False)

    # csv_to_excel("./datasets/data/estable/EXTRA/EXTRA.csv", "./datasets/data/estable_excel/EXTRA/EXTRA.xlsx")


    # Cargar el dataset combinado si es necesario
    df = pd.read_csv('./datasets/data/estable/EXTRA/EXTRA.csv')

    # Llamar a la función para plotear
    #plot_precision_by_method(df)


    # promedio = calcular_promedio_height_precision(df, 300, 400, 'RAFT-STEREO')







    

