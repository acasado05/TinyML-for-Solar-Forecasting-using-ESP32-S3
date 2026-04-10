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
corr_matrix = data_selected.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlación')
plt.savefig('modelos_tfg/correlacion_matriz.png', dpi=300, bbox_inches='tight')
plt.show()

# 2.5. Gráfica de la irradiancia global a lo largo del tiempo
plt.figure(figsize=(12, 5))
plt.plot(data_selected.index, data_selected['G_Glob'], color='orange', alpha=0.5, label='Irradiancia 10 min')
plt.title('Distribución Anual de la Irradiancia Global (G_Glob)', fontsize=16)
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Irradiancia (W/m²)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, which='major', linestyle='-', linewidth=1.2, color='black', alpha=0.3)
plt.gcf().autofmt_xdate() 
plt.tight_layout()
plt.savefig('modelos_tfg/irradiancia_anual.png', dpi=300) # Guardar en alta calidad para el TFG
plt.show()