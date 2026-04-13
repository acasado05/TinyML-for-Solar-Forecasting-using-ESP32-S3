import numpy as np
import pandas as pd
import seaborn as sns
import os
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, SimpleRNN, LSTM,GRU, Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. CARGA, PROCESADO Y LIMPIEZA DE DATOS
try:
    csv = 'modelos_tfg/datos_10min_modelos.csv'
    data = pd.read_csv(csv, sep=';', decimal=',')
    print(f"DATOS CARGADOS CORRECTAMENTE")
except FileNotFoundError:
    print(f"ERROR: El archivo '{csv}' no se encontró. Verifica la ruta y el nombre del archivo.") 

print(f"DATASET CARGADO EXITOSAMENTE:\n{data.head()}")

# 2. PREPARACIÓN DE LOS DATOS PARA EL MODELO

# - Aqui tengo pensado eliminar las filas con irradiancia = 0

# 2.1. Convertimos la columna 'Timestamp' a formato datetime y la establecemos como índice del DataFrame
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d/%m/%Y %H:%M')
data.set_index('Timestamp', inplace=True)

#2.2. Limpieza del dataset: borrado de columnas
data.drop(columns=['Gefsaypce', 'EDC', 'EACAC', 'Vmpp_panel'], inplace=True)

# 2.2. Convertimos las horas, días y meses a formato numérico cíclico
horas = data.index.hour
meses = data.index.month

data['hora_sin'] = np.sin(horas * (2 * np.pi / 24))
data['hora_cos'] = np.cos(horas * (2 * np.pi / 24))

data['mes_sin'] = np.sin(meses * (2 * np.pi / 12))
data['mes_cos'] = np.cos(meses * (2 * np.pi / 12))

# 2.3. Seleccionamos las características relevantes para el modelo
features = ['hora_sin', 'hora_cos', 'mes_sin', 'mes_cos', 'G_Glob', 'Ta', 'Hum_Rel', 'Tc', 'V_gen', 'I_gen', 'Pot_gen', 'Pot_inv'] # COMPLETAR
data_selected = data[features]

print(f"------------------------------------------------------------------------")
print(f"DATOS PREPROCESADOS:\n{data_selected.head()}")
print(f"------------------------------------------------------------------------")
print(f"ESTADÍSTICAS DESCRIPTIVAS:")
data_selected.info()
print(f"------------------------------------------------------------------------")
print(f"RESUMEN ESTADÍSTICO:\n{data_selected.describe()}")
print(f"------------------------------------------------------------------------")
print(f"Cantidad total de filas: {len(data_selected)}")
print(f"------------------------------------------------------------------------")

# 2.4. Matriz de correlación
# corr_matrix = data_selected.corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# plt.title('Matriz de Correlación')
# plt.savefig('modelos_tfg/correlacion_matriz.png', dpi=300, bbox_inches='tight')
# plt.show()

# 2.5. Gráfica de la irradiancia global a lo largo del tiempo
# plt.figure(figsize=(12, 5))
# plt.plot(data_selected.index, data_selected['G_Glob'], color='orange', alpha=0.5, label='Irradiancia 10 min')
# plt.title('Distribución Anual de la Irradiancia Global (G_Glob)', fontsize=16)
# plt.xlabel('Fecha', fontsize=12)
# plt.ylabel('Irradiancia (W/m²)', fontsize=12)
# plt.legend(loc='upper right')
# plt.grid(True, which='major', linestyle='-', linewidth=1.2, color='black', alpha=0.3)
# plt.gcf().autofmt_xdate() 
# plt.tight_layout()
# plt.savefig('modelos_tfg/irradiancia_anual.png', dpi=300) # Guardar en alta calidad para el TFG
# plt.show()

# 3. SPLIT Y NORMALIZACIÓN DE LOS DATOS

# 3.1. División de los datos en conjuntos de entrenamiento y prueba
train_split = int(0.8 * len(data_selected))
train_df = data_selected.iloc[:train_split]
val_df = data_selected.iloc[train_split:]

print(f"Entrenamiento (80%): {len(train_df)} filas")
print(f"Validación (20%): {len(val_df)} filas")

# 3.2. Normalización de los datos utilizando MinMaxScaler
scaler = MinMaxScaler()
transformed_train = scaler.fit_transform(train_df)
transformed_val = scaler.transform(val_df)

# 3.3. Función para crear secuencias de datos para el modelo
def create_multivariate_sequences(X, y, seq_length, look_ahead):
    """
    Transforma series temporales planas en ventanas deslizantes tridimensionales 
    para el entrenamiento de Redes Neuronales Recurrentes (RNN, LSTM, GRU).

    Esta función aplica la técnica de "Sliding Window" (Ventana Deslizante). 
    Recorre el dataset cronológicamente extrayendo bloques de datos históricos 
    como variables predictoras y asignando un valor futuro como objetivo.

    Parámetros:
    -----------
    X : numpy.ndarray
        Matriz bidimensional de características (features) escaladas. 
        Forma esperada: (n_muestras, n_características).
    y : numpy.ndarray
        Vector unidimensional con la variable objetivo escalada (target).
        Forma esperada: (n_muestras,).
    seq_length : int
        Tamaño de la ventana de observación (pasos de tiempo hacia el pasado).
    look_ahead : int
        Horizonte de predicción (cuántos pasos de tiempo hacia el futuro se desea predecir).

    Retorna:
    --------
    tuple
        X_seq : numpy.ndarray
            Tensores de entrada 3D con forma (muestras, seq_length, características).
        y_seq : numpy.ndarray
            Array 1D con los valores objetivo correspondientes. Forma: (muestras,).
    """
    Xs, ys = [], []

    # El bucle termina antes para evitar predecir un dato a futuro que no existe.
    limite = len(X) - seq_length - look_ahead + 1

    for i in range(limite):
        
        # Extraemos la ventana de datos históricos para la secuencia actual
        ventana_X = X[i:(i + seq_length)]

        # Apuntamos al objetivo futuro. Si la ventana termina en i + sequence_length - 1,
        # el objetivo está a 'look_ahead' pasos más allá.
        objetivo_y = y[i + seq_length + look_ahead - 1]

        Xs.append(ventana_X)
        ys.append(objetivo_y)

    return np.array(Xs), np.array(ys)

sequence_length = 18 # Ventana de 3 horas
look_ahead = 6       # Predecir 1 hora en el futuro
X_train, y_train = create_multivariate_sequences(transformed_train, transformed_train[:, -1], sequence_length, look_ahead)
X_val, y_val = create_multivariate_sequences(transformed_val, transformed_val[:, -1], sequence_length, look_ahead)

# Las GRU esperan una entrada de forma (samples, timesteps, features)
print(f"FORMA DE X: {X_train.shape}")  # (n_samples, sequence_length, n_features)
print(f"FORMA DE y: {y_train.shape}")  # (n_samples)

# 4. CREACIÓN DE LOS MODELOS DE RNN, LSTM Y GRU
def create_model(model_type, input_shape):
    
    # 1. Elegimos la capa recurrente según lo que pida la función
    if model_type == 'RNN':
        capa_recurrente = SimpleRNN(64, activation='tanh', return_sequences=False)
    elif model_type == 'LSTM':
        capa_recurrente = LSTM(64, activation='tanh', return_sequences=False)
    elif model_type == 'GRU':
        capa_recurrente = GRU(64, activation='tanh', return_sequences=False)
    
    # 2. Construimos el modelo
    model = Sequential([
        Input(shape=input_shape),
        capa_recurrente,
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1)
    ])

    # 3. Compilamos el modelo
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model
    
# ====================================================
# CREACIÓN DE LOS TRES MODELOS DE RNN (RNN, LSTM, GRU)
# ====================================================
forma_entrada = (sequence_length, X_train.shape[2])  # (18, n_features)

print(f"CONSTRUYENDO ARQUITECTURAS DE LOS MODELOS...")
model_RNN = create_model('RNN', forma_entrada)
model_LSTM = create_model('LSTM', forma_entrada)
model_GRU = create_model('GRU', forma_entrada)

model_GRU.summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4 , min_lr=1e-6, verbose=1)

n_epochs = 100
batch_size = 32

def model_training(model, model_name):

    print(f"\nINICIANDO ENTRENAMIENTO DEL MODELO {model_name}...")

    tiempo_inicio = time.time()

    history = model.fit(X_train, y_train,
                        epochs=n_epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping, reduce_lr],
                        verbose=1)
    
    tiempo_fin = time.time()
    tiempo_ejecucion = tiempo_fin - tiempo_inicio
    print(f"\nMODELO {model_name} ENTRENADO EN {tiempo_ejecucion:.2f} SEGUNDOS")

    return history

# 5. Entrenamiento de los tres modelos
print(f"\nCOMENZANDO EL ENTRENAMIENTO DE LOS MODELOS...")

historial_RNN = model_training(model_RNN, 'Simple RNN')
historial_LSTM = model_training(model_LSTM, 'LSTM')
historial_GRU = model_training(model_GRU, 'GRU')

print(f"\n ¡ENTRENAMIENTO COMPLETADO! LOS MODELOS HAN SIDO ENTRENADOS EXITOSAMENTE.")

# 6. Cálculo de métricas reales de evaluación en el conjunto de validación
print(f"\nEVALUANDO LOS MODELOS EN EL CONJUNTO DE VALIDACIÓN...")

# 6.1. Recuperamos los valores reales de potencia (Desescalar y_val)
# Creamos una matriz falsa de ceros con las 12 columnas originales
num_columnas = X_train.shape[2]
dummy_y = np.zeros((len(y_val), num_columnas))
# Metemos y_val en la última columna de la matriz dummy
dummy_y[:, -1] = y_val
# Aplicamos la inversa de la transformación para recuperar los valores reales
y_val_real = scaler.inverse_transform(dummy_y)[:, -1]

# 6.2. Función para evaluar cada modelo y calcular métricas
def evaluar_modelo(model, model_name):
    # --- A. PREDICCIÓN Y DESESCALADO ---
    predicciones = model.predict(X_val, verbose=0)
    
    dummy_pred = np.zeros((len(predicciones), num_columnas))
    dummy_pred[:, -1] = predicciones.flatten()
    predicciones_reales = scaler.inverse_transform(dummy_pred)[:, -1]
    
    # --- B. MÉTRICAS MATEMÁTICAS ---
    rmse = np.sqrt(mean_squared_error(y_val_real, predicciones_reales))
    mae = mean_absolute_error(y_val_real, predicciones_reales)
    r2 = r2_score(y_val_real, predicciones_reales)
    n = X_val.shape[0]; p = X_val.shape[2]
    r2_ajustado = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # --- C. ESTIMACIÓN DE HARDWARE (TinyML) ---
    total_params = model.count_params()
    flash_kb = (total_params * 4) / 1024
    
    # --- D. IMPRIMIMOS RESULTADOS ---
    print(f"| {model_name:10} | {mae:7.3f} kW | {rmse:7.3f} kW | {r2:6.4f} | {r2_ajustado:8.4f} | {flash_kb:8.1f} KB |")

# 6.3. Imprimimos la tabla de resultados
print("\n" + "="*83)
print(f"| {'MODELO':10} | {'MAE':10} | {'RMSE':10} | {'R^2':6} | {'R^2 Aj.':8} | {'FLASH EST.':11} |")
print("-" * 83)

evaluar_modelo(model_RNN, "Simple RNN")
evaluar_modelo(model_LSTM, "LSTM")
evaluar_modelo(model_GRU, "GRU")

print("="*83 + "\n")