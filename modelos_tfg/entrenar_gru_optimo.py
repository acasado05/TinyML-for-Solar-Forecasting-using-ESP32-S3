import numpy as np
import pandas as pd
import os
import time

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, GRU, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from tensorflow.keras.constraints import MaxNorm

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =====================================================================
# CONFIGURACIÓN DEL DIRECTORIO DE SALIDA
# =====================================================================
carpeta_salida = 'nuevo_GRU'
os.makedirs(carpeta_salida, exist_ok=True)
print(f"[*] Los resultados y el modelo se guardarán en: {carpeta_salida}")

# 1. CARGA, PROCESADO Y LIMPIEZA DE DATOS
csv = 'modelos_tfg/datos_10min_modelos.csv'
try:
    data = pd.read_csv(csv, sep=';', decimal=',')
    print(f"DATOS CARGADOS CORRECTAMENTE")
except FileNotFoundError:
    print(f"ERROR: El archivo '{csv}' no se encontró. Verifica la ruta.") 
    exit()

data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d/%m/%Y %H:%M')
data.set_index('Timestamp', inplace=True)
data.drop(columns=['Gefsaypce', 'EDC', 'EACAC', 'Vmpp_panel'], inplace=True)

horas = data.index.hour
meses = data.index.month

data['hora_sin'] = np.sin(horas * (2 * np.pi / 24))
data['hora_cos'] = np.cos(horas * (2 * np.pi / 24))
data['mes_sin'] = np.sin(meses * (2 * np.pi / 12))
data['mes_cos'] = np.cos(meses * (2 * np.pi / 12))

features = ['hora_sin', 'hora_cos', 'mes_sin', 'mes_cos', 'G_Glob', 'Ta', 'Hum_Rel', 'Tc', 'Pot_inv']
data_selected = data[features]

# 3. SPLIT Y NORMALIZACIÓN DE LOS DATOS
train_split = int(0.8 * len(data_selected))
train_df = data_selected.iloc[:train_split]
val_df = data_selected.iloc[train_split:]

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
            
    print(f"   -> Secuencias creadas: {len(Xs)} (Saltos ignorados: {saltos_ignorados})")
    return np.array(Xs), np.array(ys)

sequence_length = 18
look_ahead = 6
print("\nGenerando secuencias de entrenamiento...")
X_train, y_train = create_multivariate_sequences(transformed_train_X, transformed_train_y.flatten(), train_df.index, sequence_length, look_ahead)
print("Generando secuencias de validación...")
X_val, y_val = create_multivariate_sequences(transformed_val_X, transformed_val_y.flatten(), val_df.index, sequence_length, look_ahead)

# 4. CREACIÓN DEL MODELO GRU CON MAXNORM (Evita saturación en INT8)
K.clear_session()
input_shape = (sequence_length, X_train.shape[2])

model_GRU = Sequential([
    Input(shape=input_shape),
    # APLICACIÓN DE LAS RESTRICCIONES DE PESO PARA PROTECCIÓN INT8
    GRU(32, activation='tanh', return_sequences=False, 
        kernel_constraint=MaxNorm(2.0), 
        recurrent_constraint=MaxNorm(2.0)),
    Dropout(0.1),
    Dense(16, activation='relu'),
    Dense(1)
])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model_GRU.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# 5. ENTRENAMIENTO
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)

ruta_guardado = f"{carpeta_salida}/GRU_mejor.h5"
checkpoint = ModelCheckpoint(
    filepath=ruta_guardado, 
    monitor='val_loss', 
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

print(f"\nINICIANDO ENTRENAMIENTO DEL MODELO GRU (CON MAXNORM)...")
tiempo_inicio = time.time()

history = model_GRU.fit(X_train, y_train,
                        epochs=200,
                        batch_size=64,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping, reduce_lr, checkpoint],
                        verbose=1)

tiempo_fin = time.time()
print(f"\nMODELO GRU ENTRENADO EN {tiempo_fin - tiempo_inicio:.2f} SEGUNDOS")

# 6. EVALUACIÓN Y MÉTRICAS
print(f"\nEVALUANDO EL MODELO EN EL CONJUNTO DE VALIDACIÓN...")
y_val_real = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()

predicciones = model_GRU.predict(X_val, verbose=0)
predicciones_reales = scaler_y.inverse_transform(predicciones).flatten()
predicciones_reales = np.maximum(predicciones_reales, 0)

rmse = np.sqrt(mean_squared_error(y_val_real, predicciones_reales))
mae = mean_absolute_error(y_val_real, predicciones_reales)
r2 = r2_score(y_val_real, predicciones_reales)
n = X_val.shape[0]; p = X_val.shape[2]
r2_ajustado = 1 - (1 - r2) * (n - 1) / (n - p - 1)

total_params = model_GRU.count_params()
flash_kb = (total_params * 4) / 1024

print("\n" + "="*83)
print(f"| {'MODELO':10} | {'MAE':10} | {'RMSE':10} | {'R^2':6} | {'R^2 Aj.':8} | {'FLASH EST.':11} |")
print("-" * 83)
print(f"| {'GRU Optimo':10} | {mae:7.3f} W | {rmse:7.3f} W | {r2:6.4f} | {r2_ajustado:8.4f} | {flash_kb:8.1f} KB |")
print("="*83 + "\n")
print(f"[OK] El mejor modelo ha sido guardado en: {ruta_guardado}")