import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, GRU, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----------------------------------------------------------------------
# 1. Carga y preprocesamiento de datos
# ----------------------------------------------------------------------
# -- CARGA DEL DATASET --
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
features = ['CO(GT)', 'NO2(GT)', 'PT08.S5(O3)', 'RH', 'T', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
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

# --- ESCALADO DE LOS DATOS ---
scaler = StandardScaler()  
data_scaled = scaler.fit_transform(data_selected)

# Función para crear secuencias multivariadas: se usan 10 filas
# para predecir la siguiente fila
def create_multivariate_sequences(data, sequence_length=10):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])    # Secuencia de entrada
        y.append(data[i + sequence_length, -1])  # Valor objetivo T
    return np.array(X), np.array(y)

sequence_length = 10
X, y = create_multivariate_sequences(data_scaled, sequence_length)

# Las GRU esperan una entrada de forma (samples, timesteps, features)
print(f"FORMA DE X: {X.shape}")  # (n_samples, sequence_length, n_features)
print(f"FORMA DE y: {y.shape}")  # (n_samples)

# Asegura la forma correcta del tensor 3D al modelo GRU
X = X.reshape(X.shape[0], sequence_length, len(features)) 

# Selección de datos de entrenamiento y validación
train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]