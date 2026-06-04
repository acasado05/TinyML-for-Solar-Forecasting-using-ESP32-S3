import numpy as np
import pandas as pd
import seaborn as sns
import os
import re
import time

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, LSTM, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =====================================================================
# GUARDADO DE VERSIONES DE LAS GRÁFICAS DE VALIDACIÓN Y RESULTADOS
# =====================================================================
ruta_base = 'train_LSTM'
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
    csv = 'train_LSTM/datos_10min_utc.csv'
    data = pd.read_csv(csv, sep=';', decimal=',')
    print(f"DATOS CARGADOS CORRECTAMENTE")
except FileNotFoundError:
    print(f"ERROR: El archivo '{csv}' no se encontró. Verifica la ruta y el nombre del archivo.") 

print(f"DATASET CARGADO EXITOSAMENTE:\n{data.head()}")

# 2. PREPARACIÓN DE LOS DATOS PARA EL MODELO

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
features = ['hora_sin', 'hora_cos', 'mes_sin', 'mes_cos', 'G_Glob', 'Ta', 'Hum_Rel', 'Tc', 'Pot_inv'] 
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

train_X_df = train_df.drop(columns=['Pot_inv'])
val_X_df = val_df.drop(columns=['Pot_inv'])

transformed_train_X = scaler_X.fit_transform(train_X_df)
transformed_val_X = scaler_X.transform(val_X_df)

transformed_train_y = scaler_y.fit_transform(train_df[['Pot_inv']])
transformed_val_y = scaler_y.transform(val_df[['Pot_inv']])

# 3.3. Función para crear secuencias de datos para el modelo
def create_multivariate_sequences(X, y, timestamps, seq_length, look_ahead):
    Xs, ys = [], []
    limite = len(X) - seq_length - look_ahead + 1
    minutos_esperados = 10 * (seq_length + look_ahead - 1)
    tiempo_esperado = pd.Timedelta(minutes=minutos_esperados)
    saltos_ignorados = 0

    for i in range(limite):
        tiempo_inicio = timestamps[i]
        tiempo_fin = timestamps[i + seq_length + look_ahead - 1]
        tiempo_real = tiempo_fin - tiempo_inicio

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

print(f"FORMA DE X: {X_train.shape}")  
print(f"FORMA DE y: {y_train.shape}")  

# 4. CREACIÓN DEL MODELO LSTM
def create_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(32, activation='tanh', return_sequences=False),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model
    
forma_entrada = (sequence_length, X_train.shape[2])  

K.clear_session()  

print(f"CONSTRUYENDO ARQUITECTURA DEL MODELO LSTM...")
model_LSTM = create_lstm_model(forma_entrada)

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
        save_best_only=True,   
        save_weights_only=False, 
        verbose=1
    )

    tiempo_inicio = time.time()
    history = model.fit(X_train, y_train,
                        epochs=n_epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping, reduce_lr, checkpoint],
                        verbose=1)
    
    tiempo_ejecucion = time.time() - tiempo_inicio
    print(f"\nMODELO {model_name} ENTRENADO EN {tiempo_ejecucion:.2f} SEGUNDOS")
    return history

# 5. Entrenamiento del modelo
print(f"\nCOMENZANDO EL ENTRENAMIENTO DEL MODELO...")
historial_LSTM = model_training(model_LSTM, 'LSTM')
print(f"\n ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")

# 6. Evaluación en validación
print(f"\nEVALUANDO EL MODELO EN EL CONJUNTO DE VALIDACIÓN...")
y_val_real = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()

def evaluar_modelo(model, model_name):
    predicciones = model.predict(X_val, verbose=0)
    predicciones_reales = scaler_y.inverse_transform(predicciones).flatten()
    predicciones_reales = np.maximum(predicciones_reales, 0)
    
    rmse = np.sqrt(mean_squared_error(y_val_real, predicciones_reales))
    mae = mean_absolute_error(y_val_real, predicciones_reales)
    r2 = r2_score(y_val_real, predicciones_reales)
    n = X_val.shape[0]; p = X_val.shape[2]
    r2_ajustado = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    total_params = model.count_params()
    flash_kb = (total_params * 4) / 1024
    
    print(f"| {model_name:10} | {mae:7.3f} W | {rmse:7.3f} W | {r2:6.4f} | {r2_ajustado:8.4f} | {flash_kb:8.1f} KB |")
    return predicciones_reales, mae, r2, flash_kb

print("\n" + "="*83)
print(f"| {'MODELO':10} | {'MAE':10} | {'RMSE':10} | {'R^2':6} | {'R^2 Aj.':8} | {'FLASH EST.':11} |")
print("-" * 83)
preds_lstm_real, mae_lstm, r2_lstm, kb_lstm = evaluar_modelo(model_LSTM, "LSTM")
print("="*83 + "\n")

# 7. Visualizaciones adaptadas para LSTM

COLOR_REAL = '#000000'
COLOR_LSTM = '#1976D2'

# GRÁFICA 1: Evolución Val_Loss
plt.figure(figsize=(10, 5))
plt.plot(historial_LSTM.history['val_loss'], label='LSTM', color=COLOR_LSTM, linewidth=2.5)
plt.title('Evolución del Error de Validación durante el Entrenamiento', fontsize=16, fontweight='bold')
plt.xlabel('Épocas', fontsize=13)
plt.ylabel('Loss (MSE)', fontsize=13)
plt.legend(fontsize=12)
plt.grid(True, which='major', linestyle='--', linewidth=1.2, color='black', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{carpeta_salida}/1_val_loss_lstm.png', dpi=300)
plt.show()

# GRÁFICA 1B: Train vs Val Loss
plt.figure(figsize=(10, 5))
loss = historial_LSTM.history['loss']
val_loss = historial_LSTM.history['val_loss']
epocas = range(len(loss))
plt.plot(epocas, loss, label='Training Loss', color='gray', linestyle='--', linewidth=2)
plt.plot(epocas, val_loss, label='Validation Loss', color=COLOR_LSTM, linewidth=2.5)
plt.title('Diagnóstico de Entrenamiento: Pérdida (Train) vs Validación (Val)', fontsize=16, fontweight='bold')
plt.xlabel('Épocas', fontsize=13)
plt.ylabel('Loss (MSE)', fontsize=13)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'{carpeta_salida}/1b_diagnostico_train_val.png', dpi=300)
plt.show()

# GRÁFICA 2: Dispersión
plt.figure(figsize=(8, 8))
max_val = np.max(y_val_real) * 1.05
plt.scatter(y_val_real, preds_lstm_real, alpha=0.6, color=COLOR_LSTM, s=20, label='Predicciones LSTM')
plt.plot([0, max_val], [0, max_val], color=COLOR_REAL, linestyle='--', linewidth=2.5, label='Ideal')
plt.title('Dispersión del Modelo LSTM', fontsize=16, fontweight='bold')
plt.xlabel('Potencia Real (W)', fontsize=13)
plt.ylabel('Potencia Predicha (W)', fontsize=13)
plt.xlim(0, max_val)
plt.ylim(0, max_val)
plt.legend(fontsize=12)
plt.grid(True, which='major', linestyle='--', linewidth=1.2, color='black', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{carpeta_salida}/2_dispersion.png', dpi=300)
plt.show()

# GRÁFICA 3: Día Soleado
DIA_SOLEADO_INICIO = 0
DIA_SOLEADO_FIN = 144
plt.figure(figsize=(14, 5))
plt.plot(y_val_real[DIA_SOLEADO_INICIO:DIA_SOLEADO_FIN], label='Real', color=COLOR_REAL, linewidth=3.5)
plt.plot(preds_lstm_real[DIA_SOLEADO_INICIO:DIA_SOLEADO_FIN], label='LSTM', color=COLOR_LSTM, linewidth=2)
plt.title('Detalle de Predicción: Día Despejado (Curva de Campana)', fontsize=16, fontweight='bold')
plt.xlabel('Pasos de Tiempo (10 min)', fontsize=13)
plt.ylabel('Potencia (W)', fontsize=13)
plt.legend(fontsize=12)
plt.grid(True, which='major', linestyle='--', linewidth=1.2, color='black', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{carpeta_salida}/3_zoom_soleado.png', dpi=300)
plt.show()

# GRÁFICA 4: Día Nublado
DIA_NUBLADO_INICIO = 4170
DIA_NUBLADO_FIN = 4320
plt.figure(figsize=(14, 5))
plt.plot(y_val_real[DIA_NUBLADO_INICIO:DIA_NUBLADO_FIN], label='Real', color=COLOR_REAL, linewidth=3.5)
plt.plot(preds_lstm_real[DIA_NUBLADO_INICIO:DIA_NUBLADO_FIN], label='LSTM', color=COLOR_LSTM, linewidth=2)
plt.title('Detalle de Predicción: Día Nublado (Alta Variabilidad)', fontsize=16, fontweight='bold')
plt.xlabel('Pasos de Tiempo (10 min)', fontsize=13)
plt.ylabel('Potencia (W)', fontsize=13)
plt.legend(fontsize=12)
plt.grid(True, which='major', linestyle='--', linewidth=1.2, color='black', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{carpeta_salida}/4_zoom_nublado.png', dpi=300)
plt.show()

# GRÁFICA 5: Comparativa Temporal
INICIO = 280
FIN = 1300
plt.figure(figsize=(16, 6))
plt.plot(y_val_real[INICIO:FIN], label='Potencia Real Medida', color=COLOR_REAL, linewidth=3.5, zorder=6)
plt.plot(preds_lstm_real[INICIO:FIN], label='LSTM', color=COLOR_LSTM, linewidth=2, alpha=0.9)
plt.title('Comparativa de Potencia Generada (Finales Octubre)', fontsize=16, fontweight='bold')
plt.xlabel('Pasos de Tiempo (Intervalos de 10 min)', fontsize=13)
plt.ylabel('Potencia (W)', fontsize=13)
plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
plt.grid(True, which='major', linestyle='--', linewidth=1.2, color='black', alpha=0.3)
plt.margins(x=0)
plt.tight_layout()
plt.savefig(f'{carpeta_salida}/5_comparativa_temporal_kW.png', dpi=300)
plt.show()

# GRÁFICA 6: Barras Bi-objetivo LSTM
etiquetas = ['LSTM']
valores_mae = [mae_lstm]
valores_kb = [kb_lstm]

x = np.arange(len(etiquetas))
width = 0.35  

fig, ax1 = plt.subplots(figsize=(8, 6))

bar1 = ax1.bar(x - width/2, valores_mae, width, label='Error MAE (W)', color='#F57C00', edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Error Promedio (MAE en W)', fontsize=13, fontweight='bold', color='#F57C00')
ax1.tick_params(axis='y', labelcolor='#F57C00', labelsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(etiquetas, fontsize=13, fontweight='bold')
ax1.set_ylim(0, max(valores_mae) * 1.5) 

ax2 = ax1.twinx()  
bar2 = ax2.bar(x + width/2, valores_kb, width, label='Tamaño Estimado (KB)', color='#4527A0', edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Memoria Flash Estimada (KB)', fontsize=13, fontweight='bold', color='#4527A0')
ax2.tick_params(axis='y', labelcolor='#4527A0', labelsize=11)
ax2.set_ylim(0, max(valores_kb) * 1.5)

plt.title('Precisión vs. Ligereza de Hardware (LSTM)', fontsize=16, fontweight='bold', pad=15)

for bar in bar1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + (max(valores_mae)*0.05), 
             f'{yval:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

for bar in bar2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + (max(valores_kb)*0.05), 
             f'{yval:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(top=0.85) 
plt.savefig(f'{carpeta_salida}/6_barras_biobjetivo.png', dpi=300)
plt.show()