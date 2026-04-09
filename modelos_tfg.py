import numpy as np
import pandas as pd
import seaborn as sns
import os

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
    csv = 'datos_10min.csv'
    data = pd.read_csv(csv, sep=',', decimal=',')
    print(f"DATOS CARGADOS CORRECTAMENTE")
except FileNotFoundError:
    print(f"ERROR: El archivo '{csv}' no se encontró. Verifica la ruta y el nombre del archivo.") 

print(f"DATASET CARGADO EXITOSAMENTE:\n{data.head()}")

# 2. PREPARACIÓN DE LOS DATOS PARA EL MODELO

# - Aqui tengo pensado eliminar las filas con irradiancia = 0

# 2.1. Convertimos la columna 'Timestamp' a formato datetime y la establecemos como índice del DataFrame
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d/%m/%Y %H:%M')
data.set_index('Timestamp', inplace=True)

# 2.2. Convertimos las horas, días y meses a formato numérico cíclico
horas = data.index.hour
meses = data.index.month

data['hora_sin'] = np.sin(horas * (2 * np.pi / 24))
data['hora_cos'] = np.cos(horas * (2 * np.pi / 24))

data['mes_sin'] = np.sin(meses * (2 * np.pi / 12))
data['mes_cos'] = np.cos(meses * (2 * np.pi / 12))

# 2.3. Seleccionamos las características relevantes para el modelo
features = ['hora_sin', 'hora_cos', 'mes_sin', 'mes_cos', 'Irradiancia', 'Temperatura', 'Humedad'] # COMPLETAR
data_selected = data[features]

print(f"DATOS PREPROCESADOS:\n{data_selected.head()}")
print(f"ESTADÍSTICAS DESCRIPTIVAS:\n{data_selected.info()}")
print(f"RESUMEN ESTADÍSTICO:\n{data_selected.describe()}")
print(f"Cantidad total de filas: {len(data_selected)}")
