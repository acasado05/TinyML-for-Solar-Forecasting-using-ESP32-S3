import numpy as np
import pandas as pd
import seaborn as sns
import os
import re
import time

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, SimpleRNN, LSTM,GRU, Dense, Dropout
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
features = ['hora_sin', 'hora_cos', 'mes_sin', 'mes_cos', 'G_Glob', 'Ta', 'Hum_Rel', 'Tc', 'V_gen', 'I_gen', 'Pot_gen', 'Pot_inv']
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

# # 2.5. Matriz de correlación
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

sequence_length = 30 # Ventana de 5 horas
look_ahead = 6       # Predecir 1 hora en el futuro
X_train, y_train = create_multivariate_sequences(transformed_train_X, transformed_train_y.flatten(), train_df.index, sequence_length, look_ahead)
X_val, y_val = create_multivariate_sequences(transformed_val_X, transformed_val_y.flatten(), val_df.index, sequence_length, look_ahead)

# Las GRU esperan una entrada de forma (samples, timesteps, features)
print(f"FORMA DE X: {X_train.shape}")  # (n_samples, sequence_length, n_features)
print(f"FORMA DE y: {y_train.shape}")  # (n_samples)

# 4. CREACIÓN DE LOS MODELOS DE RNN, LSTM Y GRU
def create_model(model_type, input_shape):
    
    # 1. Elegimos la capa recurrente según lo que pida la función
    if model_type == 'RNN':
        model = Sequential([
            Input(shape=input_shape),
            SimpleRNN(64, activation='tanh', return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='relu')
        ])
        optimizer = keras.optimizers.Adam(learning_rate=0.001)

    elif model_type == 'LSTM':
        model = Sequential([
            Input(shape=input_shape),
            LSTM(32, activation='tanh', return_sequences=True),
            Dropout(0.1),
            LSTM(16, activation='tanh', return_sequences=False),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1, activation='relu')
        ])
        optimizer = keras.optimizers.Adam(learning_rate=0.0005)

    elif model_type == 'GRU':
        model = Sequential([
            Input(shape=input_shape),
            GRU(32, activation='tanh', return_sequences=True),
            Dropout(0.1),
            GRU(16, activation='tanh', return_sequences=False),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1, activation='relu')
        ])
        optimizer = keras.optimizers.Adam(learning_rate=0.0005)

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

model_GRU.summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7 , min_lr=1e-6, verbose=1)

n_epochs = 100
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

print("="*83 + "\n")

# 7. Visualizaciones de los resultados de los entrenamientos

# Constantes de colores para las gráficas
COLOR_REAL = '#000000'
COLOR_RNN  = '#D32F2F'
COLOR_LSTM = '#1976D2'
COLOR_GRU  = '#388E3C'

# =====================================================================
# GRÁFICA 1: LA CARRERA DEL APRENDIZAJE (Val Loss de los 3 juntos)
# =====================================================================
plt.figure(figsize=(10, 6))
plt.plot(historial_RNN.history['val_loss'], label='Simple RNN', color=COLOR_RNN, linewidth=2.5)
plt.plot(historial_LSTM.history['val_loss'], label='LSTM', color=COLOR_LSTM, linewidth=2.5)
plt.plot(historial_GRU.history['val_loss'], label='GRU', color=COLOR_GRU, linewidth=2.5)

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
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Diagnóstico de Entrenamiento: Pérdida (Train) vs Validación (Val)', fontsize=16, fontweight='bold')

# Historiales y nombres para iterar fácilmente
historiales = [historial_RNN, historial_LSTM, historial_GRU]
nombres = ['Simple RNN', 'LSTM', 'GRU']
colores = [COLOR_RNN, COLOR_LSTM, COLOR_GRU]

for i in range(3):
    ax = axes[i]
    # Extraemos los datos de la memoria del entrenamiento
    loss = historiales[i].history['loss']
    val_loss = historiales[i].history['val_loss']
    epocas = range(len(loss))
    
    # Dibujamos Entrenamiento (línea normal) y Validación (línea gruesa del color del modelo)
    ax.plot(epocas, loss, label='Training Loss', color='gray', linestyle='--', linewidth=2)
    ax.plot(epocas, val_loss, label='Validation Loss', color=colores[i], linewidth=2.5)
    
    ax.set_title(f'Modelo: {nombres[i]}', fontsize=14)
    ax.set_xlabel('Épocas', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
# Ajustamos un poco para que el título principal no pise las gráficas
plt.subplots_adjust(top=0.88) 
plt.savefig(f'{carpeta_salida}/1b_diagnostico_train_val.png', dpi=300)
plt.show()

# =====================================================================
# GRÁFICA 2: DISPERSIÓN DEL GANADOR
# =====================================================================
# 1. Empaquetamos la información de los modelos en un diccionario
modelos_info = {
    'Simple RNN': {'preds': preds_rnn_real, 'r2': r2_rnn, 'color': COLOR_RNN},
    'LSTM':       {'preds': preds_lstm_real, 'r2': r2_lstm, 'color': COLOR_LSTM},
    'GRU':        {'preds': preds_gru_real,  'r2': r2_gru,  'color': COLOR_GRU}
}

# 2. Buscamos el modelo con el R^2 más alto
mejor_nombre = max(modelos_info, key=lambda k: modelos_info[k]['r2'])
mejor_preds  = modelos_info[mejor_nombre]['preds']
mejor_color  = modelos_info[mejor_nombre]['color']
mejor_r2     = modelos_info[mejor_nombre]['r2']

print(f"\n Modelo seleccionado para la Gráfica de Dispersión: {mejor_nombre} (R^2 = {mejor_r2:.4f})")

# 3. Gráfica de dispersión del modelo ganador
plt.figure(figsize=(8, 8))
max_val = np.max(y_val_real) * 1.05

plt.figure(figsize=(8, 8))
max_val = np.max(y_val_real) * 1.05

plt.scatter(y_val_real, mejor_preds, alpha=0.6, color=mejor_color, s=20, label=f'Predicciones {mejor_nombre}')
plt.plot([0, max_val], [0, max_val], color=COLOR_REAL, linestyle='--', linewidth=2.5, label='Ideal')

plt.title(f'Dispersión del Modelo Óptimo ({mejor_nombre}): Real vs. Predicción', fontsize=16, fontweight='bold')
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

plt.figure(figsize=(12, 5))
plt.plot(y_val_real[DIA_SOLEADO_INICIO:DIA_SOLEADO_FIN], label='Real', color=COLOR_REAL, linewidth=3.5)
plt.plot(preds_rnn_real[DIA_SOLEADO_INICIO:DIA_SOLEADO_FIN], label='RNN', color=COLOR_RNN, linewidth=2, linestyle='--')
plt.plot(preds_lstm_real[DIA_SOLEADO_INICIO:DIA_SOLEADO_FIN], label='LSTM', color=COLOR_LSTM, linewidth=2)
plt.plot(preds_gru_real[DIA_SOLEADO_INICIO:DIA_SOLEADO_FIN], label='GRU', color=COLOR_GRU, linewidth=2)

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

plt.figure(figsize=(12, 5))
plt.plot(y_val_real[DIA_NUBLADO_INICIO:DIA_NUBLADO_FIN], label='Real', color=COLOR_REAL, linewidth=3.5)
plt.plot(preds_rnn_real[DIA_NUBLADO_INICIO:DIA_NUBLADO_FIN], label='RNN', color=COLOR_RNN, linewidth=2, linestyle='--')
plt.plot(preds_lstm_real[DIA_NUBLADO_INICIO:DIA_NUBLADO_FIN], label='LSTM', color=COLOR_LSTM, linewidth=2)
plt.plot(preds_gru_real[DIA_NUBLADO_INICIO:DIA_NUBLADO_FIN], label='GRU', color=COLOR_GRU, linewidth=2)

plt.title('Detalle de Predicción: Día Nublado (Alta Variabilidad)', fontsize=16, fontweight='bold')
plt.xlabel('Pasos de Tiempo (10 min)', fontsize=13)
plt.ylabel('Potencia (W)', fontsize=13)
plt.legend(fontsize=12)
plt.grid(True, which='major', linestyle='--', linewidth=1.2, color='black', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{carpeta_salida}/4_zoom_nublado.png', dpi=300)
plt.show()

# =====================================================================
# GRÁFICA 5: COMPARATIVA TEMPORAL EN kW (Real vs. Predicciones)
# =====================================================================
INICIO = 280
FIN = 1300

plt.figure(figsize=(16, 6))

# Dibujamos la línea de potencia REAL (más gruesa para que destaque)
plt.plot(y_val_real[INICIO:FIN], label='Potencia Real Medida', color=COLOR_REAL, linewidth=3.5, zorder=5)

# Dibujamos las predicciones de los modelos
plt.plot(preds_rnn_real[INICIO:FIN], label='Predicción RNN', color=COLOR_RNN, linewidth=2, linestyle='--', alpha=0.9)
plt.plot(preds_lstm_real[INICIO:FIN], label='Predicción LSTM', color=COLOR_LSTM, linewidth=2, alpha=0.9)
plt.plot(preds_gru_real[INICIO:FIN], label='Predicción GRU', color=COLOR_GRU, linewidth=2, alpha=0.9)

plt.title('Comparativa de Potencia Generada (Finales Octubre)', fontsize=16, fontweight='bold')
plt.xlabel('Pasos de Tiempo (Intervalos de 10 min)', fontsize=13)
plt.ylabel('Potencia (W)', fontsize=13)

# Ponemos la leyenda fuera del gráfico si tapa las curvas, o ajustamos su posición
plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
plt.grid(True, which='major', linestyle='--', linewidth=1.2, color='black', alpha=0.3)

# Ajustes para que no se corte nada al guardar
plt.margins(x=0)
plt.tight_layout()

# Guardamos la imagen
plt.savefig(f'{carpeta_salida}/5_comparativa_temporal_kW.png', dpi=300)
plt.show()

# =====================================================================
# GRÁFICA 6: BARRAS BI-OBJETIVO (Error MAE vs Tamaño en KB)
# =====================================================================
# Empaquetamos los valores que ya calculaste previamente en la tabla
etiquetas = ['Simple RNN', 'LSTM', 'GRU']
valores_mae = [mae_rnn, mae_lstm, mae_gru]
valores_kb = [kb_rnn, kb_lstm, kb_gru]

x = np.arange(len(etiquetas))
width = 0.35  # Ancho de las barras

# Creamos la figura "A lo grande"
fig, ax1 = plt.subplots(figsize=(10, 6))

# --- EJE Y IZQUIERDO (MAE) ---
# Usamos un naranja fuerte y marcamos el borde en negro (ideal para imprimir)
bar1 = ax1.bar(x - width/2, valores_mae, width, label='Error MAE (kW)', color='#F57C00', edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Error Promedio (MAE en W)', fontsize=13, fontweight='bold', color='#F57C00')
ax1.tick_params(axis='y', labelcolor='#F57C00', labelsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(etiquetas, fontsize=13, fontweight='bold')

# Damos un 15% de espacio extra arriba para que quepan los números sin pisar el borde
ax1.set_ylim(0, max(valores_mae) * 1.15) 

# --- EJE Y DERECHO (KILOBYTES) ---
# Clonamos el eje X para crear un segundo eje Y
ax2 = ax1.twinx()  
# Usamos un morado/índigo fuerte
bar2 = ax2.bar(x + width/2, valores_kb, width, label='Tamaño Estimado (KB)', color='#4527A0', edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Memoria Flash Estimada (KB)', fontsize=13, fontweight='bold', color='#4527A0')
ax2.tick_params(axis='y', labelcolor='#4527A0', labelsize=11)

# Damos un 15% de espacio extra arriba también aquí
ax2.set_ylim(0, max(valores_kb) * 1.15)

# --- DETALLES DE FORMATO ---
plt.title('Comparativa TinyML: Precisión vs. Ligereza de Hardware', fontsize=16, fontweight='bold', pad=15)


# ¡EL TRUCO PRO!: Añadir los valores exactos encima de cada barra
# Esto evita que el tribunal tenga que usar una regla para "adivinar" el valor
for bar in bar1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + (max(valores_mae)*0.02), 
             f'{yval:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

for bar in bar2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + (max(valores_kb)*0.02), 
             f'{yval:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
# Bajamos un poco el área del gráfico para hacer hueco a la leyenda superior
plt.subplots_adjust(top=0.82) 

# Guardamos en alta resolución
plt.savefig(f'{carpeta_salida}/6_barras_biobjetivo.png', dpi=300)
plt.show()