import numpy as np
import pandas as pd
import seaborn as sns
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, GRU, Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. CARGA, PROCESADO Y LIMPIEZA DE DATOS
try:
    csv = 'AirQualityUCI.csv'
    data = pd.read_csv(csv, sep=';', decimal=',')
    print(f"DATOS CARGADOS CORRECTAMENTE")
except FileNotFoundError:
    print(f"ERROR: El archivo '{csv}' no se encontró. Asegúrate de que el archivo esté en el directorio correcto.")

print(f"ESTRUCTURA DE LOS DATOS:\n{data.head()}")

# -- LIMPIEZA DE DATOS --
data.replace(-200, np.nan, inplace=True)  # Reemplaza valores faltantes

# Combina fecha y hora en una sola columna de tipo datetime
data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H.%M.%S')

# Elimina las columnas originales de fecha y hora
data.drop(['Date', 'Time'], axis=1, inplace=True)

# Elimina columnas innecesarias
data.drop(columns=['Unnamed: 15', 'Unnamed: 16'], inplace=True)

# En las filas especificadas, si hay valores NaN, elimina esas filas
data.dropna(subset=['CO(GT)', 'NO2(GT)', 'T', 'RH'], inplace=True)

# Con ffill rellena los huecos usando la historia del sensor (datos anteriores) y con bfill rellena los huecos usando el futuro del sensor (datos posteriores)
data.ffill(inplace=True)
data.bfill(inplace=True)

# Set DateTime as index: convierte a esa columna en una línea temporal
data.set_index('DateTime', inplace=True)

# 2. Ingeniería de características y análisis exploratorio

# Añado variables cíclicas para la hora del día, dia y mes
data['hour'] = data.index.hour + data.index.minute / 60.0
data['month'] = data.index.month

# Hora (Periodo = 24)
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

# Mes (Periodo = 12)
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

# Selecciona solamente los parámetros que voy a querer de entrada para el modelo
features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos','CO(GT)', 'NO2(GT)', 'PT08.S5(O3)', 'RH', 'T']
data_selected = data[features]

print(f"DATOS PREPROCESADOS:\n{data_selected.head()}")
print(f"ESTADÍSTICAS DESCRIPTIVAS:\n{data_selected.info()}")
print(f"RESUMEN ESTADÍSTICO:\n{data_selected.describe()}")

# Matriz de correlación
corr_matrix = data_selected.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlación')
plt.savefig('correlacion_matriz.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. SPLIT Y NORMALIZACIÓN DE LOS DATOS
# 1º Dividimos y selección de datos de entrenamiento y validación (80%/20%)
train_split = int(0.8 * len(data_selected))
train_df = data_selected.iloc[:train_split]
test_df = data_selected.iloc[train_split:]

# 2º NORMALIZACIÓN DE LOS DATOS
scaler = MinMaxScaler()  
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

# 4. CREACIÓN DE SECUENCIAS PARA EL MODELO GRU
# Función para crear secuencias multivariadas: se usan 10 filas
# para predecir la siguiente fila
def create_multivariate_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])    # Secuencia de entrada
        y.append(data[i + sequence_length, -1])  # Valor objetivo T
    return np.array(X), np.array(y)

sequence_length = 15
X_train, y_train = create_multivariate_sequences(train_scaled, sequence_length)
X_val, y_val = create_multivariate_sequences(test_scaled, sequence_length)
# Nota: X ya sale con la forma (n_samples, sequence_length, n_features) gracias a la función creada, 
# pero se asegura la forma correcta para el modelo GRU. Por lo que no hace falta el reshape manual posterior
# Si los datos fuesen univariados, si que me haría falta un reshape para convertirlos a formato 3D, 
# pero al ser multivariados, ya salen con la forma correcta. PONER RESHAPE ES REDUNDANTE Y PUEDE CAUSAR ERRORES DE FORMA SI NO SE HACE CORRECTAMENTE.

# Las GRU esperan una entrada de forma (samples, timesteps, features)
print(f"FORMA DE X: {X_train.shape}")  # (n_samples, sequence_length, n_features)
print(f"FORMA DE y: {y_train.shape}")  # (n_samples)

# Asegura la forma correcta del tensor 3D al modelo GRU -> NO HACE FALTA RESHAPE
# X_train = X_train.reshape(X_train.shape[0], sequence_length, len(features)) 
# X_val = X_val.reshape(X_val.shape[0], sequence_length, len(features)) 

# 5. CONSTRUCCIÓN DEL MODELO GRU
model = Sequential([Input(shape=(sequence_length, len(features))),
                    GRU(32, activation='tanh', return_sequences=False),
                    Dropout(0.1), #20% de dropout para evitar overfitting
                    #Dense(32, activation='relu'), # Se aumenta la densidad para V2
                    Dense(16, activation='relu'),
                    Dense(8, activation='relu'), # Se añade una capa adicional para V3 para limpiar la salida
                    Dense(1)  # Capa de salida para regresión
                    ])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

model.summary()

# 6. ENTRENAMIENTO DEL MODELO
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4 , min_lr=1e-6, verbose=1) 
checkpoint = keras.callbacks.ModelCheckpoint('best_gru_aq_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

print("INICIANDO EL ENTRENAMIENTO...")
history = model.fit(X_train, y_train, 
                    epochs=70, 
                    batch_size=32, 
                    validation_data=(X_val, y_val), 
                    callbacks=[early_stopping, reduce_lr, checkpoint],
                    verbose=1)

#model.save('gru_aq_model.h5')
#print("MODELO GUARDADO COMO 'gru_aq_model.h5'")

print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

# 7. PREDICCIÓN Y EVALUACIÓN DEL MODELO
y_pred_scaled = model.predict(X_val)
y_pred_scaled = np.clip(y_pred_scaled, 0, 1)  # Limita las predicciones a un rango razonable

def get_original_scale(scaled_y, scaler, n_features):
    # Creamos una matriz de ceros con el mismo número de columnas que el dataset original (5)
    dummy_matrix = np.zeros((len(scaled_y), n_features))
    
    # Colocamos nuestros valores (predichos o reales) en la última columna, 
    # que es donde estaba la Temperatura 'T' en el scaler.
    dummy_matrix[:, -1] = scaled_y.flatten()
    
    # Aplicamos la inversión del escalado
    unscaled_matrix = scaler.inverse_transform(dummy_matrix)
    
    # Extraemos solo la columna de la temperatura (la última)
    return unscaled_matrix[:, -1]

# 2. Aplicamos la función a las predicciones y a los valores reales
y_pred_rescaled = get_original_scale(y_pred_scaled, scaler, len(features))
y_true_rescaled = get_original_scale(y_val, scaler, len(features))

# 3. Evaluamos el rendimiento del modelo con las métricas originales
rmse = np.sqrt(mean_squared_error(y_true_rescaled, y_pred_rescaled))
mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
r2 = r2_score(y_true_rescaled, y_pred_rescaled)

# Calculo del R^2 ajustado
n = X_val.shape[0]  # Número de muestras
p = X_val.shape[2] # Número de predictores (en este caso, el número de sensores)
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R^2: {r2:.4f}")
print(f"R^2 ajustado: {r2_adj:.4f}")

# 8. BLOQUE DE GRÁFICAS PROFESIONALES
plt.figure(figsize=(20, 6))
output_folder = 'gráficas_modelo_v1'
os.makedirs(output_folder, exist_ok=True)

# --- 8.1 GRÁFICA DE CONVERGENCIA (LOSS) ---
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss', color='#1f77b4', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
plt.title('Convergencia del Modelo (MSE)', fontsize=14, fontweight='bold')
plt.xlabel('Época')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Si training_loss >> validation_loss, es posible que el modelo esté subentrenado (underfitting).
# Si training_loss << validation_loss, es posible que el modelo esté sobreentrenado (overfitting).

# --- 8.2 GRÁFICA DE DISPERSIÓN Y REGRESIÓN ---
plt.subplot(1, 3, 2)
sns.scatterplot(x=y_true_rescaled, y=y_pred_rescaled, alpha=0.5, color='#2ca02c')
# Línea de identidad (45 grados)
lims = [min(min(y_true_rescaled), min(y_pred_rescaled)), 
        max(max(y_true_rescaled), max(y_pred_rescaled))]
plt.plot(lims, lims, color='red', linestyle='--', label='Identidad (y=x)')
plt.title(f'Dispersión: Real vs Predicho\n(R² = {r2:.4f})', fontsize=14, fontweight='bold')
plt.xlabel('Valores Reales (°C)')
plt.ylabel('Predicciones (°C)')
plt.legend()

# --- 8.3 HISTOGRAMA DE RESIDUOS (ERROR) ---
plt.subplot(1, 3, 3)
residuos = y_true_rescaled - y_pred_rescaled
sns.histplot(residuos, kde=True, color='#d62728', bins=30)
plt.axvline(x=0, color='black', linestyle='--')
plt.title('Distribución de Residuos (Errores)', fontsize=14, fontweight='bold')
plt.xlabel('Error (Real - Predicho)')
plt.ylabel('Frecuencia')

plt.tight_layout()
plt.savefig(f'{output_folder}/modelo_v1_temp.png', dpi=300) # Guardar en alta calidad

# ==========================================
# 9. COMPARATIVA TEMPORAL (ZOOM AL TEST)
# ==========================================
plt.figure(figsize=(15, 6))

# Mostramos un fragmento (ej. las últimas 150 horas) para que se aprecie el detalle
plt.plot(y_true_rescaled[-150:], label='Temperatura Real', color='#1f77b4', linewidth=2, alpha=0.8)
plt.plot(y_pred_rescaled[-150:], label='Predicción GRU', color='#d62728', linestyle='--', linewidth=2)

plt.title('Seguimiento Temporal: Real vs Predicción (Últimas 150h)', fontsize=14, fontweight='bold')
plt.xlabel('Tiempo (Horas en el conjunto de Test)')
plt.ylabel('Temperatura (°C)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# Guardamos en tu nueva carpeta de gráficas
plt.savefig(f'{output_folder}/temp_real_vs_pred_v1.png', dpi=300, bbox_inches='tight')
plt.show()

# Instrucción para obtener el número de parámetros
total_params = model.count_params()

# Cálculo en bytes
size_bytes = total_params * 4
size_kb = size_bytes / 1024

print(f"Total de parámetros: {total_params}")
print(f"Tamaño teórico en RAM: {size_bytes} bytes ({size_kb:.2f} KB)")