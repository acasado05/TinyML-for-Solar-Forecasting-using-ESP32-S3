import numpy as np
import pandas as pd
import seaborn as sns
import os
import re
import time

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =====================================================================
# GUARDADO DE VERSIONES DE LAS GRÁFICAS DE VALIDACIÓN Y RESULTADOS
# =====================================================================
ruta_base = 'modelos_tfg'
prefijo = 'entrenamiento_MLP_v'
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

# 2. PREPARACIÓN DE LOS DATOS PARA EL MODELO
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d/%m/%Y %H:%M')
data.set_index('Timestamp', inplace=True)

data.drop(columns=['Gefsaypce', 'EDC', 'EACAC', 'Vmpp_panel'], inplace=True)

horas = data.index.hour
meses = data.index.month

data['hora_sin'] = np.sin(horas * (2 * np.pi / 24))
data['hora_cos'] = np.cos(horas * (2 * np.pi / 24))
data['mes_sin'] = np.sin(meses * (2 * np.pi / 12))
data['mes_cos'] = np.cos(meses * (2 * np.pi / 12))

# Sin V_gen ni I_gen
features = ['hora_sin', 'hora_cos', 'mes_sin', 'mes_cos', 'G_Glob', 'Ta', 'Hum_Rel', 'Tc', 'Pot_inv'] 
data_selected = data[features]

print(f"------------------------------------------------------------------------")
print(f"Cantidad total de filas: {len(data_selected)}")
print(f"------------------------------------------------------------------------")

# 3. SPLIT Y NORMALIZACIÓN DE LOS DATOS
train_split = int(0.8 * len(data_selected))
train_df = data_selected.iloc[:train_split]
val_df = data_selected.iloc[train_split:]

print(f"Entrenamiento (80%): {len(train_df)} filas")
print(f"Validación (20%): {len(val_df)} filas")

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

train_X_df = train_df.drop(columns=['Pot_inv'])
val_X_df = val_df.drop(columns=['Pot_inv'])

transformed_train_X = scaler_X.fit_transform(train_X_df)
transformed_val_X = scaler_X.transform(val_X_df)

transformed_train_y = scaler_y.fit_transform(train_df[['Pot_inv']])
transformed_val_y = scaler_y.transform(val_df[['Pot_inv']])

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

    print(f"   -> Secuencias creadas: {len(Xs)} (Ignoradas: {saltos_ignorados})")
    return np.array(Xs), np.array(ys)

sequence_length = 6    # 3 horas
look_ahead = 6         # Predecir 1 hora en el futuro
X_train, y_train = create_multivariate_sequences(transformed_train_X, transformed_train_y.flatten(), train_df.index, sequence_length, look_ahead)
X_val, y_val = create_multivariate_sequences(transformed_val_X, transformed_val_y.flatten(), val_df.index, sequence_length, look_ahead)

# =====================================================================
# 3.4. FILTRADO DE NOCHES (Aislado para el MLP)
# =====================================================================
print("\n[INFO] Filtrando periodos nocturnos para el modelo MLP...")
umbral = 0.01 

mascara_sol_train = y_train > umbral
X_train_dia = X_train[mascara_sol_train]
y_train_dia = y_train[mascara_sol_train]

mascara_sol_val = y_val > umbral
X_val_dia = X_val[mascara_sol_val]
y_val_dia = y_val[mascara_sol_val]

print(f"   -> Train: {len(X_train)} originales -> {len(X_train_dia)} diurnas")
print(f"   -> Val:   {len(X_val)} originales -> {len(X_val_dia)} diurnas\n")

# 4. CREACIÓN DEL MODELO MLP
def create_mlp_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Flatten(), # Aplasta las secuencias 2D a 1D
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1) # Linear por defecto (sin ReLU para evitar dying ReLU)
    ])
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

forma_entrada = (sequence_length, X_train.shape[2]) 
K.clear_session() 

print(f"CONSTRUYENDO ARQUITECTURA DEL MLP...")
model_MLP = create_mlp_model(forma_entrada)
model_MLP.summary()

# 5. ENTRENAMIENTO DEL MODELO MLP
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7 , min_lr=1e-6, verbose=1)

n_epochs = 200
batch_size = 64

print(f"\nINICIANDO ENTRENAMIENTO DEL MODELO MLP (SOLO DE DÍA)...")
ruta_guardado = f"{carpeta_salida}/MLP_mejor.h5"
checkpoint = ModelCheckpoint(filepath=ruta_guardado, monitor='val_loss', save_best_only=True, verbose=1)

tiempo_inicio = time.time()
historial_MLP = model_MLP.fit(X_train_dia, y_train_dia,
                              epochs=n_epochs,
                              batch_size=batch_size,
                              validation_data=(X_val_dia, y_val_dia),
                              callbacks=[early_stopping, reduce_lr, checkpoint],
                              verbose=1)
tiempo_fin = time.time()
print(f"\nMODELO MLP ENTRENADO EN {(tiempo_fin - tiempo_inicio):.2f} SEGUNDOS")

# 6. EVALUACIÓN CON EL DATASET DE VALIDACIÓN COMPLETO (DÍA + NOCHE)
print(f"\nEVALUANDO EL MODELO MLP EN EL CONJUNTO DE VALIDACIÓN COMPLETO...")
y_val_real = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()

# Predicción sobre el val COMPLETO (X_val) no el filtrado (X_val_dia)
predicciones = model_MLP.predict(X_val, verbose=0)
predicciones_reales = scaler_y.inverse_transform(predicciones).flatten()

# CLIPPING FÍSICO: Eliminamos potencias negativas
predicciones_reales = np.maximum(predicciones_reales, 0)

rmse = np.sqrt(mean_squared_error(y_val_real, predicciones_reales))
mae = mean_absolute_error(y_val_real, predicciones_reales)
r2 = r2_score(y_val_real, predicciones_reales)
n = X_val.shape[0]; p = X_val.shape[2]
r2_ajustado = 1 - (1 - r2) * (n - 1) / (n - p - 1)

total_params = model_MLP.count_params()
flash_kb = (total_params * 4) / 1024

print("\n" + "="*83)
print(f"| {'MODELO':10} | {'MAE':10} | {'RMSE':10} | {'R^2':6} | {'R^2 Aj.':8} | {'FLASH EST.':11} |")
print("-" * 83)
print(f"| {'MLP':10} | {mae:7.3f} W | {rmse:7.3f} W | {r2:6.4f} | {r2_ajustado:8.4f} | {flash_kb:8.1f} KB |")
print("="*83 + "\n")

# 7. VISUALIZACIONES (Solo para MLP)
COLOR_REAL = '#000000'
COLOR_MLP  = '#FF9800' # Naranja

# 7.1 Curvas de Pérdida
plt.figure(figsize=(8, 5))
loss = historial_MLP.history['loss']
val_loss = historial_MLP.history['val_loss']
epocas = range(len(loss))
plt.plot(epocas, loss, label='Training Loss (Día)', color='gray', linestyle='--', linewidth=2)
plt.plot(epocas, val_loss, label='Validation Loss (Día)', color=COLOR_MLP, linewidth=2.5)
plt.title('Diagnóstico de Entrenamiento: MLP', fontsize=14, fontweight='bold')
plt.xlabel('Épocas', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'{carpeta_salida}/1_diagnostico_train_val.png', dpi=300)
plt.show()

# 7.2 Dispersión
plt.figure(figsize=(7, 7))
max_val = np.max(y_val_real) * 1.05
plt.scatter(y_val_real, predicciones_reales, alpha=0.6, color=COLOR_MLP, s=20, label='Predicciones MLP')
plt.plot([0, max_val], [0, max_val], color=COLOR_REAL, linestyle='--', linewidth=2.5, label='Ideal')
plt.title('Dispersión del Modelo MLP: Real vs. Predicción', fontsize=14, fontweight='bold')
plt.xlabel('Potencia Real (W)', fontsize=12)
plt.ylabel('Potencia Predicha (W)', fontsize=12)
plt.xlim(0, max_val)
plt.ylim(0, max_val)
plt.legend(fontsize=11)
plt.grid(True, which='major', linestyle='--', linewidth=1.2, color='black', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{carpeta_salida}/2_dispersion.png', dpi=300)
plt.show()

# 7.3 Zoom Soleado
DIA_SOLEADO_INICIO = 0
DIA_SOLEADO_FIN = 144
plt.figure(figsize=(12, 4))
plt.plot(y_val_real[DIA_SOLEADO_INICIO:DIA_SOLEADO_FIN], label='Real', color=COLOR_REAL, linewidth=3.5)
plt.plot(predicciones_reales[DIA_SOLEADO_INICIO:DIA_SOLEADO_FIN], label='MLP', color=COLOR_MLP, linewidth=2)
plt.title('Detalle de Predicción: Día Despejado (MLP)', fontsize=14, fontweight='bold')
plt.xlabel('Pasos de Tiempo (10 min)', fontsize=12)
plt.ylabel('Potencia (W)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, which='major', linestyle='--', linewidth=1.2, color='black', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{carpeta_salida}/3_zoom_soleado.png', dpi=300)
plt.show()

# 7.4 Zoom Nublado
DIA_NUBLADO_INICIO = 4170
DIA_NUBLADO_FIN = 4320
plt.figure(figsize=(12, 4))
plt.plot(y_val_real[DIA_NUBLADO_INICIO:DIA_NUBLADO_FIN], label='Real', color=COLOR_REAL, linewidth=3.5)
plt.plot(predicciones_reales[DIA_NUBLADO_INICIO:DIA_NUBLADO_FIN], label='MLP', color=COLOR_MLP, linewidth=2)
plt.title('Detalle de Predicción: Día Nublado (MLP)', fontsize=14, fontweight='bold')
plt.xlabel('Pasos de Tiempo (10 min)', fontsize=12)
plt.ylabel('Potencia (W)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, which='major', linestyle='--', linewidth=1.2, color='black', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{carpeta_salida}/4_zoom_nublado.png', dpi=300)
plt.show()

# 7.5 Comparativa Temporal Final
INICIO = 280
FIN = 1300
plt.figure(figsize=(16, 5))
plt.plot(y_val_real[INICIO:FIN], label='Potencia Real', color=COLOR_REAL, linewidth=3.5, zorder=6)
plt.plot(predicciones_reales[INICIO:FIN], label='Predicción MLP', color=COLOR_MLP, linewidth=2, alpha=0.9)
plt.title('Comparativa Temporal de Potencia Generada (MLP)', fontsize=14, fontweight='bold')
plt.xlabel('Pasos de Tiempo', fontsize=12)
plt.ylabel('Potencia (W)', fontsize=12)
plt.legend(fontsize=11, loc='upper right', framealpha=0.9)
plt.grid(True, which='major', linestyle='--', linewidth=1.2, color='black', alpha=0.3)
plt.margins(x=0)
plt.tight_layout()
plt.savefig(f'{carpeta_salida}/5_comparativa_temporal.png', dpi=300)
plt.show()