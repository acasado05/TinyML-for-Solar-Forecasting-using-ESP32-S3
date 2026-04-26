import numpy as np
import pandas as pd
import seaborn as sns
import os
import re
import time

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, SimpleRNN, LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =====================================================================
# GUARDADO DE VERSIONES DE LAS GRÁFICAS DE VALIDACIÓN Y RESULTADOS
# =====================================================================
ruta_base = 'modelos_tfg'
prefijo = 'entrenamiento_v'
os.makedirs(ruta_base, exist_ok=True)
carpetas_existentes = [d for d in os.listdir(ruta_base) if os.path.isdir(os.path.join(ruta_base, d)) and d.startswith(prefijo)]

if not carpetas_existentes:
    nueva_version = 1
else:
    numeros = []
    for c in carpetas_existentes:
        try:
            num = int(re.findall(r'^' + prefijo + r'(\d+)', c)[0])
            numeros.append(num)
        except (IndexError, ValueError):
            continue

    nueva_version = max(numeros) + 1 if numeros else 1

carpeta_salida = os.path.join(ruta_base, f'{prefijo}{nueva_version}')
os.makedirs(carpeta_salida, exist_ok=True)

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

# 2.3. Convertimos las horas, días y meses a formato numérico cíclico
horas = data.index.hour
meses = data.index.month

data['hora_sin'] = np.sin(horas * (2 * np.pi / 24))
data['hora_cos'] = np.cos(horas * (2 * np.pi / 24))

data['mes_sin'] = np.sin(meses * (2 * np.pi / 12))
data['mes_cos'] = np.cos(meses * (2 * np.pi / 12))

# 2.4. Seleccionamos las características relevantes para el modelo
features = ['hora_sin', 'hora_cos', 'mes_sin', 'mes_cos', 'G_Glob', 'Ta', 'Hum_Rel', 'Tc', 'Pot_inv'] # Quito Pot_Gen, V_gen e I_gen
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

# 2.5. Matriz de correlación
# corr_matrix = data_selected.corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# plt.title('Matriz de Correlación')
# plt.savefig('modelos_tfg/correlacion_matriz.png', dpi=300, bbox_inches='tight')
# plt.show()

# # 2.6. Gráfica de la irradiancia global a lo largo del tiempo
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
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Separamos las variables predictoras (X) de la variable objetivo (y)
train_X_df = train_df.drop(columns=['Pot_inv'])
val_X_df = val_df.drop(columns=['Pot_inv'])

# Escalamos X (Todas las columnas menos Pot_inv)
transformed_train_X = scaler_X.fit_transform(train_X_df)
transformed_val_X = scaler_X.transform(val_X_df)

# Escalamos y (Solo Pot_inv)
transformed_train_y = scaler_y.fit_transform(train_df[['Pot_inv']])
transformed_val_y = scaler_y.transform(val_df[['Pot_inv']])

# 3.3. Función para crear secuencias de datos para el modelo
def create_multivariate_sequences(X, y, timestamps, seq_length, look_ahead):
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

    # El bucle termina antes para evitar salirnos del límite
    limite = len(X) - seq_length - look_ahead + 1

    # Calculamos el tiempo exacto que DEBE haber si no hay saltos (en minutos)
    # Ejemplo: (18 + 6 - 1) * 10 min = 230 minutos exactos
    minutos_esperados = 10 * (seq_length + look_ahead - 1)
    tiempo_esperado = pd.Timedelta(minutes=minutos_esperados)

    saltos_ignorados = 0

    for i in range(limite):
        
        # Miramos la fecha de inicio de la ventana y la fecha del objetivo
        tiempo_inicio = timestamps[i]
        tiempo_fin = timestamps[i + seq_length + look_ahead - 1]
        
        # Calculamos cuánto tiempo ha pasado realmente
        tiempo_real = tiempo_fin - tiempo_inicio

        # Solo guardamos la secuencia si el tiempo es EXACTO (no faltan filas en medio)
        if tiempo_real == tiempo_esperado:
            ventana_X = X[i:(i + seq_length)]
            objetivo_y = y[i + seq_length + look_ahead - 1]

            Xs.append(ventana_X)
            ys.append(objetivo_y)
        else:
            saltos_ignorados += 1

    print(f"   -> Secuencias creadas: {len(Xs)} (Se ignoraron {saltos_ignorados} secuencias por saltos temporales)")

    return np.array(Xs), np.array(ys)

sequence_length = 18   # 3 horas
look_ahead = 6         # Predecir 1 hora en el futuro
X_train, y_train = create_multivariate_sequences(transformed_train_X, transformed_train_y.flatten(), train_df.index, sequence_length, look_ahead)
X_val, y_val = create_multivariate_sequences(transformed_val_X, transformed_val_y.flatten(), val_df.index, sequence_length, look_ahead)

# Las GRU esperan una entrada de forma (samples, timesteps, features)
print(f"FORMA DE X: {X_train.shape}")  # (n_samples, sequence_length, n_features)
print(f"FORMA DE y: {y_train.shape}")  # (n_samples)

# # 4. CREACIÓN DE LOS MODELOS DE RNN, LSTM Y GRU
def create_model(model_type, input_shape):
    
    # 1. Elegimos la capa recurrente según lo que pida la función
    if model_type == 'RNN':
        model = Sequential([
            Input(shape=input_shape),
            SimpleRNN(64, activation='tanh', return_sequences=False),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        optimizer = keras.optimizers.Adam(learning_rate=0.001)

    elif model_type == 'LSTM':
        model = Sequential([
            Input(shape=input_shape),
            LSTM(32, activation='tanh', return_sequences=False),
            Dropout(0.1),
            # LSTM(16, activation='tanh', return_sequences=False),
            #Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        optimizer = keras.optimizers.Adam(learning_rate=0.001)

    elif model_type == 'GRU':
        model = Sequential([
            Input(shape=input_shape),
            GRU(32, activation='tanh', return_sequences=False),
            Dropout(0.1),
            #GRU(16, activation='tanh', return_sequences=False),
            #Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    elif model_type == 'CNN':
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    elif model_type == 'MLP':
        model = Sequential([
            Input(shape=input_shape),  # Solo características, sin dimensión temporal
            Flatten(),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        optimizer = keras.optimizers.Adam(learning_rate=0.001)

    # 2. Compilamos el modelo
    
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model
    
# ====================================================
# CREACIÓN DE LOS TRES MODELOS DE RNN (RNN, LSTM, GRU)
# ====================================================
forma_entrada = (sequence_length, X_train.shape[2])  # (18, n_features)

K.clear_session()  # Limpiamos la sesión para evitar conflictos con modelos anteriores

print(f"CONSTRUYENDO ARQUITECTURAS DE LOS MODELOS...")
model_RNN = create_model('RNN', forma_entrada)
model_LSTM = create_model('LSTM', forma_entrada)
model_GRU = create_model('GRU', forma_entrada)
model_CNN = create_model('CNN', forma_entrada)
model_MLP = create_model('MLP', forma_entrada)

#model_GRU.summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7 , min_lr=1e-6, verbose=1)

n_epochs = 200
batch_size = 64

def model_training(model, model_name):

    print(f"\nINICIANDO ENTRENAMIENTO DEL MODELO {model_name}...")

    nombre_archivo = f"{model_name.replace(' ', '_')}_mejor.h5"
    ruta_guardado = f"{carpeta_salida}/{nombre_archivo}"

    checkpoint = ModelCheckpoint(
        filepath=ruta_guardado, 
        monitor='val_loss', 
        save_best_only=True,   # Solo guarda el mejor, no guarda épocas malas
        save_weights_only=False, # Guarda el modelo completo (arquitectura + pesos)
        verbose=1
    )

    tiempo_inicio = time.time()

    history = model.fit(X_train, y_train,
                        epochs=n_epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping, reduce_lr, checkpoint],
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
historial_CNN = model_training(model_CNN, 'CNN')
historial_MLP = model_training(model_MLP, 'MLP')

print(f"\n ¡ENTRENAMIENTO COMPLETADO! LOS MODELOS HAN SIDO ENTRENADOS EXITOSAMENTE.")

# 6. Cálculo de métricas reales de evaluación en el conjunto de validación
print(f"\nEVALUANDO LOS MODELOS EN EL CONJUNTO DE VALIDACIÓN...")

# 6.1. Recuperamos los valores reales de potencia (Desescalar y_val)
y_val_real = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()

# 6.2. Función para evaluar cada modelo y calcular métricas
def evaluar_modelo(model, model_name):
    # --- A. PREDICCIÓN Y DESESCALADO ---
    predicciones = model.predict(X_val, verbose=0)
    
    # Inversa directa limpia
    predicciones_reales = scaler_y.inverse_transform(predicciones).flatten()
    # Convertimos cualquier valor negativo predicho por la red en un 0 absoluto.
    predicciones_reales = np.maximum(predicciones_reales, 0)
    
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
    print(f"| {model_name:10} | {mae:7.3f} W | {rmse:7.3f} W | {r2:6.4f} | {r2_ajustado:8.4f} | {flash_kb:8.1f} KB |")

    return predicciones_reales, mae, r2, flash_kb

# 6.3. Imprimimos la tabla de resultados
print("\n" + "="*83)
print(f"| {'MODELO':10} | {'MAE':10} | {'RMSE':10} | {'R^2':6} | {'R^2 Aj.':8} | {'FLASH EST.':11} |")
print("-" * 83)

preds_rnn_real, mae_rnn, r2_rnn, kb_rnn = evaluar_modelo(model_RNN, "Simple RNN")
preds_lstm_real, mae_lstm, r2_lstm, kb_lstm = evaluar_modelo(model_LSTM, "LSTM")
preds_gru_real, mae_gru, r2_gru, kb_gru = evaluar_modelo(model_GRU, "GRU")
preds_cnn_real, mae_cnn, r2_cnn, kb_cnn = evaluar_modelo(model_CNN, "CNN")
preds_mlp_real, mae_mlp, r2_mlp, kb_mlp = evaluar_modelo(model_MLP, "MLP")
print("="*83 + "\n")

# 7. Visualizaciones de los resultados de los entrenamientos

# Constantes de colores para las gráficas
COLOR_REAL = '#000000'
COLOR_RNN  = '#D32F2F'
COLOR_LSTM = '#1976D2'
COLOR_GRU  = '#388E3C'
COLOR_CNN  = '#9C27B0'
COLOR_MLP  = '#FF9800'

# =====================================================================
# GRÁFICA 1: LA CARRERA DEL APRENDIZAJE
# =====================================================================
plt.figure(figsize=(12, 6))
plt.plot(historial_RNN.history['val_loss'], label='Simple RNN', color=COLOR_RNN, linewidth=2.5)
plt.plot(historial_LSTM.history['val_loss'], label='LSTM', color=COLOR_LSTM, linewidth=2.5)
plt.plot(historial_GRU.history['val_loss'], label='GRU', color=COLOR_GRU, linewidth=2.5)
plt.plot(historial_CNN.history['val_loss'], label='CNN', color=COLOR_CNN, linewidth=2.5)
plt.plot(historial_MLP.history['val_loss'], label='MLP', color=COLOR_MLP, linewidth=2.5)

plt.title('Evolución del Error de Validación durante el Entrenamiento', fontsize=16, fontweight='bold')
plt.xlabel('Épocas', fontsize=13)
plt.ylabel('Loss (MSE)', fontsize=13)
plt.legend(fontsize=12)
plt.grid(True, which='major', linestyle='--', linewidth=1.2, color='black', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{carpeta_salida}/1_val_loss_unificado.png', dpi=300)
plt.show()

# =====================================================================
# GRÁFICA 1B: SALUD DEL MODELO (Train vs Validation Loss)
# =====================================================================
# Como ahora son 5 modelos, usamos una cuadrícula de 3x2 y ocultamos el último panel
fig, axes = plt.subplots(3, 2, figsize=(16, 15))
fig.suptitle('Diagnóstico de Entrenamiento: Pérdida (Train) vs Validación (Val)', fontsize=16, fontweight='bold')

historiales = [historial_RNN, historial_LSTM, historial_GRU, historial_CNN, historial_MLP]
nombres = ['Simple RNN', 'LSTM', 'GRU', 'CNN', 'MLP']
colores = [COLOR_RNN, COLOR_LSTM, COLOR_GRU, COLOR_CNN, COLOR_MLP]
axes = axes.flatten()

for i in range(5):
    ax = axes[i]
    loss = historiales[i].history['loss']
    val_loss = historiales[i].history['val_loss']
    epocas = range(len(loss))
    
    ax.plot(epocas, loss, label='Training Loss', color='gray', linestyle='--', linewidth=2)
    ax.plot(epocas, val_loss, label='Validation Loss', color=colores[i], linewidth=2.5)
    
    ax.set_title(f'Modelo: {nombres[i]}', fontsize=14)
    ax.set_xlabel('Épocas', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)

# Ocultar el 6º panel vacío
fig.delaxes(axes[5])

plt.tight_layout()
plt.subplots_adjust(top=0.93) 
plt.savefig(f'{carpeta_salida}/1b_diagnostico_train_val.png', dpi=300)
plt.show()

# =====================================================================
# GRÁFICA 2: DISPERSIÓN DEL GANADOR
# =====================================================================
modelos_info = {
    'Simple RNN': {'preds': preds_rnn_real, 'r2': r2_rnn, 'color': COLOR_RNN},
    'LSTM':       {'preds': preds_lstm_real, 'r2': r2_lstm, 'color': COLOR_LSTM},
    'GRU':        {'preds': preds_gru_real,  'r2': r2_gru,  'color': COLOR_GRU},
    'CNN':        {'preds': preds_cnn_real,  'r2': r2_cnn,  'color': COLOR_CNN},
    'MLP':        {'preds': preds_mlp_real,  'r2': r2_mlp,  'color': COLOR_MLP}
}

mejor_nombre = max(modelos_info, key=lambda k: modelos_info[k]['r2'])
mejor_preds  = modelos_info[mejor_nombre]['preds']
mejor_color  = modelos_info[mejor_nombre]['color']
mejor_r2     = modelos_info[mejor_nombre]['r2']

print(f"\n Modelo seleccionado para la Gráfica de Dispersión: {mejor_nombre} (R^2 = {mejor_r2:.4f})")

plt.figure(figsize=(8, 8))
max_val = np.max(y_val_real) * 1.05

plt.scatter(y_val_real, mejor_preds, alpha=0.6, color=mejor_color, s=20, label=f'Predicciones {mejor_nombre}')
plt.plot([0, max_val], [0, max_val], color=COLOR_REAL, linestyle='--', linewidth=2.5, label='Ideal')

plt.title(f'Dispersión del Modelo Óptimo ({mejor_nombre})', fontsize=16, fontweight='bold')
plt.xlabel('Potencia Real (W)', fontsize=13)
plt.ylabel('Potencia Predicha (W)', fontsize=13)
plt.xlim(0, max_val)
plt.ylim(0, max_val)
plt.legend(fontsize=12)
plt.grid(True, which='major', linestyle='--', linewidth=1.2, color='black', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{carpeta_salida}/2_dispersion_ganador.png', dpi=300)
plt.show()

# =====================================================================
# GRÁFICA 3: ZOOM DÍA SOLEADO INDIVIDUAL
# =====================================================================
DIA_SOLEADO_INICIO = 0
DIA_SOLEADO_FIN = 144

plt.figure(figsize=(14, 5))
plt.plot(y_val_real[DIA_SOLEADO_INICIO:DIA_SOLEADO_FIN], label='Real', color=COLOR_REAL, linewidth=3.5)
plt.plot(preds_rnn_real[DIA_SOLEADO_INICIO:DIA_SOLEADO_FIN], label='RNN', color=COLOR_RNN, linewidth=2, linestyle='--')
plt.plot(preds_lstm_real[DIA_SOLEADO_INICIO:DIA_SOLEADO_FIN], label='LSTM', color=COLOR_LSTM, linewidth=2)
plt.plot(preds_gru_real[DIA_SOLEADO_INICIO:DIA_SOLEADO_FIN], label='GRU', color=COLOR_GRU, linewidth=2)
plt.plot(preds_cnn_real[DIA_SOLEADO_INICIO:DIA_SOLEADO_FIN], label='CNN', color=COLOR_CNN, linewidth=2)
plt.plot(preds_mlp_real[DIA_SOLEADO_INICIO:DIA_SOLEADO_FIN], label='MLP', color=COLOR_MLP, linewidth=2)

plt.title('Detalle de Predicción: Día Despejado (Curva de Campana)', fontsize=16, fontweight='bold')
plt.xlabel('Pasos de Tiempo (10 min)', fontsize=13)
plt.ylabel('Potencia (W)', fontsize=13)
plt.legend(fontsize=12)
plt.grid(True, which='major', linestyle='--', linewidth=1.2, color='black', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{carpeta_salida}/3_zoom_soleado.png', dpi=300)
plt.show()

# =====================================================================
# GRÁFICA 4: ZOOM DÍA NUBLADO INDIVIDUAL
# =====================================================================
DIA_NUBLADO_INICIO = 4170
DIA_NUBLADO_FIN = 4320

plt.figure(figsize=(14, 5))
plt.plot(y_val_real[DIA_NUBLADO_INICIO:DIA_NUBLADO_FIN], label='Real', color=COLOR_REAL, linewidth=3.5)
plt.plot(preds_rnn_real[DIA_NUBLADO_INICIO:DIA_NUBLADO_FIN], label='RNN', color=COLOR_RNN, linewidth=2, linestyle='--')
plt.plot(preds_lstm_real[DIA_NUBLADO_INICIO:DIA_NUBLADO_FIN], label='LSTM', color=COLOR_LSTM, linewidth=2)
plt.plot(preds_gru_real[DIA_NUBLADO_INICIO:DIA_NUBLADO_FIN], label='GRU', color=COLOR_GRU, linewidth=2)
plt.plot(preds_cnn_real[DIA_NUBLADO_INICIO:DIA_NUBLADO_FIN], label='CNN', color=COLOR_CNN, linewidth=2)
plt.plot(preds_mlp_real[DIA_NUBLADO_INICIO:DIA_NUBLADO_FIN], label='MLP', color=COLOR_MLP, linewidth=2)

plt.title('Detalle de Predicción: Día Nublado (Alta Variabilidad)', fontsize=16, fontweight='bold')
plt.xlabel('Pasos de Tiempo (10 min)', fontsize=13)
plt.ylabel('Potencia (W)', fontsize=13)
plt.legend(fontsize=12)
plt.grid(True, which='major', linestyle='--', linewidth=1.2, color='black', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{carpeta_salida}/4_zoom_nublado.png', dpi=300)
plt.show()

# =====================================================================
# GRÁFICA 5: COMPARATIVA TEMPORAL EN kW
# =====================================================================
INICIO = 280
FIN = 1300

plt.figure(figsize=(16, 6))

plt.plot(y_val_real[INICIO:FIN], label='Potencia Real Medida', color=COLOR_REAL, linewidth=3.5, zorder=6)
plt.plot(preds_rnn_real[INICIO:FIN], label='RNN', color=COLOR_RNN, linewidth=2, linestyle='--', alpha=0.9)
plt.plot(preds_lstm_real[INICIO:FIN], label='LSTM', color=COLOR_LSTM, linewidth=2, alpha=0.9)
plt.plot(preds_gru_real[INICIO:FIN], label='GRU', color=COLOR_GRU, linewidth=2, alpha=0.9)
plt.plot(preds_cnn_real[INICIO:FIN], label='CNN', color=COLOR_CNN, linewidth=2, alpha=0.9)
plt.plot(preds_mlp_real[INICIO:FIN], label='MLP', color=COLOR_MLP, linewidth=2, alpha=0.9)

plt.title('Comparativa de Potencia Generada (Finales Octubre)', fontsize=16, fontweight='bold')
plt.xlabel('Pasos de Tiempo (Intervalos de 10 min)', fontsize=13)
plt.ylabel('Potencia (W)', fontsize=13)

plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
plt.grid(True, which='major', linestyle='--', linewidth=1.2, color='black', alpha=0.3)
plt.margins(x=0)
plt.tight_layout()
plt.savefig(f'{carpeta_salida}/5_comparativa_temporal_kW.png', dpi=300)
plt.show()

# =====================================================================
# GRÁFICA 6: BARRAS BI-OBJETIVO
# =====================================================================
etiquetas = ['Simple RNN', 'LSTM', 'GRU', 'CNN', 'MLP']
valores_mae = [mae_rnn, mae_lstm, mae_gru, mae_cnn, mae_mlp]
valores_kb = [kb_rnn, kb_lstm, kb_gru, kb_cnn, kb_mlp]

x = np.arange(len(etiquetas))
width = 0.35  

fig, ax1 = plt.subplots(figsize=(12, 6))

bar1 = ax1.bar(x - width/2, valores_mae, width, label='Error MAE (W)', color='#F57C00', edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Error Promedio (MAE en W)', fontsize=13, fontweight='bold', color='#F57C00')
ax1.tick_params(axis='y', labelcolor='#F57C00', labelsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(etiquetas, fontsize=13, fontweight='bold')
ax1.set_ylim(0, max(valores_mae) * 1.15) 

ax2 = ax1.twinx()  
bar2 = ax2.bar(x + width/2, valores_kb, width, label='Tamaño Estimado (KB)', color='#4527A0', edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Memoria Flash Estimada (KB)', fontsize=13, fontweight='bold', color='#4527A0')
ax2.tick_params(axis='y', labelcolor='#4527A0', labelsize=11)
ax2.set_ylim(0, max(valores_kb) * 1.15)

plt.title('Comparativa TinyML: Precisión vs. Ligereza de Hardware', fontsize=16, fontweight='bold', pad=15)

for bar in bar1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + (max(valores_mae)*0.02), 
             f'{yval:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

for bar in bar2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + (max(valores_kb)*0.02), 
             f'{yval:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(top=0.82) 
plt.savefig(f'{carpeta_salida}/6_barras_biobjetivo.png', dpi=300)
plt.show()