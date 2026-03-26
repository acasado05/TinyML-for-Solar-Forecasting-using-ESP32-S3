#Data and numerical operations
import pandas as pd
import numpy as np

#Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dense

#Visualization
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Carga y normalización de datos. Carga el dataset de Air Quality desde la UR.
url = "https://codesignal-staging-assets.s3.amazonaws.com/uploads/1742293523899/AirQualityUCI.csv"
df = pd.read_csv(url, sep=';', decimal=',')

# Sustituye datos con valor -200 por NaN
df.replace(-200, np.nan, inplace=True) 

# Combina las columnas 'Date' y 'Time' en una sola columna de fecha y hora
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')

# Elimina las columnas 'Date' y 'Time' originales
df.drop(columns=['Date', 'Time'], inplace=True)

# Elimina columnas innecesarias
df.drop(columns=['Unnamed: 15', 'Unnamed: 16'], inplace=True)

# En las filas especificadas, si hay valos NaN, elimina esas filas
df.dropna(subset=['CO(GT)','NO2(GT)','T','RH'], inplace=True)

# Con ffill rellena los huecos usando la historia del sensor (datos anteriores) y con bfill rellena los huecos usando el futuro del sensor (datos posteriores)
df.ffill(inplace=True)
df.bfill(inplace=True)

# Set DateTime as index: convierte a esa columna en una línea temporal
df.set_index('DateTime', inplace=True)

# Selecciona parámetros importantes para la RNN (GRU)
features = ['CO(GT)', 'NO2(GT)', 'PT08.S5(O3)' ,'RH', 'T']
df_selected = df[features] # Crea un nuevo dataframe con solo las columnas seleccionadas

# Normaliza los datos con StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

# Función que transforma una tabla plana en cubos de información para la GRU.
# data: es el array de datos normalizados (5 columnas de los sensores)
# sequence_length: es el número de filas que se usarán para predecir la siguiente fila. 
# En este caso, se usarán 10 filas para predecir la siguiente fila.
def create_multivariate_sequences(data, sequence_length=10):
    X, y = [], []
    for i in range(len(data) - sequence_length): # Recorre todo el dataset, pero se detiene 10 posiciones antes del final para no salir del rango al crear las secuencias
        X.append(data[i:i+sequence_length])      # La secuencia de entrada es un bloque de 10 filas consecutivas del dataset + 5 columnas de sensores
        y.append(data[i+sequence_length, -1])    # Es la última columna (la temperatura). El modelo intenta predecir la temperatura de la siguiente fila usando las 10 filas anteriores.
    return np.array(X), np.array(y)

# Define la longitud de la secuencia
sequence_length = 10

# Esta línea llama a la función anterior para crear las secuencias de entrada:
# (X) es un bloque de datos de los 5 sensores durante 10 horas
# (y) es donde se guarda el valor que se quiere predecir para la hora 11 de cada ventana
X, y = create_multivariate_sequences(df_scaled, sequence_length)

# Las GRU exigen tensores de 3 dimensiones: (n_samples, sequence_length, n_features)
# - n_samples: número total de ventanas para entrenar al modelo. En este caso, es el número total de filas del dataset menos 10 (porque cada ventana usa 10 filas).
# - sequence_length: número de timestamps que se usarán para predecir la siguiente fila. En este caso, es 10.
# - len(features): número de sensores que se están usando como input para la GRU. En este caso, es 5 (CO, NO2, PT08.S5, RH y T).
# Esta línea asegura que el formato sea como el que pide la librería keras para entrenar la GRU. Si no se hace esto, el modelo no entenderá los datos y dará error.
X = X.reshape((X.shape[0], sequence_length, len(features))) # Asegura que X tenga la forma correcta para la GRU: (n_samples, sequence_length, n_features)

# Split data into training and testing sets (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Para construir es necesario importar tensorflow, y luego API de keras para models y layers. En este caso, hay dos modelos GRUs.
# Son varias capas. El input es la puerta de entrada. Va a recibir un bloque de 10 horas seguidas. Y en cada hora, hay 5 datos.
model = Sequential([Input(shape=(sequence_length, len(features))), # Define la forma de los datos de entrada para la GRU
                    GRU(32, activation='tanh', return_sequences=True), # Primera capa GRU con 32 unidades. Con return_sequences=True para que devuelva una secuencia de 
                                                                       #salida que se pueda usar como input para la siguiente capa GRU.
                    GRU(16, activation='tanh'), # Segunda capa GRU con 16 unidades. No devuelve secuencias porque es la última capa GRU.
                    Dense(1) # Capa densa final con 1 unidad para predecir la temperatura de la siguiente hora. Obtiene lo que soltó la anterior GRU para predecir la temperatura.
                    ])

# Con el modelo definido, podemos compilarlo y entrenarlo. Para compilarlo, se define el optimizador (Adam) y la función de pérdida (MSE). Luego, se entrena el modelo con los datos de entrada (X) y las etiquetas (y) durante 20 épocas.
model.compile(optimizer='adam', loss='mse') # Compila el modelo con el optimizador Adam y la función de pérdida MSE (error cuadrático medio)

# Pasa por parámetro los datos de entrada (X) y las etiquetas (y) para entrenar el modelo durante 10 épocas con un tamaño de lote de 16
#model.fit(X, y, epochs=10, batch_size=16)

# Train the model - Note: This will cause an error until you compile the model correctly
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

model.summary()

# Print final loss values
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

# Aqui el modelo hace predicciones con los datos de prueba (X_test) para evaluar su rendimiento. 
# El resultado se guarda en la variable y_pred, que contiene las predicciones de temperatura para 
# las horas siguientes a las ventanas de 10 horas usadas como input.
y_pred = model.predict(X_test)

# Reescala las predicciones y los valores actuales a los valores originales. Crea una matriz llena de ceros. Tiene tantas filas
# como predicciones (len(y_pred)) y tantas columnas como sensores menos 1 (len(features) - 1). Luego, se combina esta matriz de 
# ceros con las predicciones (y_pred) para que tengan la misma forma que los datos originales. Finalmente, se aplica la función 
# inverse_transform del scaler para reescalar los valores a su rango original, y se selecciona solo la primera columna (la temperatura) 
# para obtener las predicciones reescaladas.
#y_pred_rescaled = scaler.inverse_transform(np.column_stack((y_pred, np.zeros((len(y_pred), len(features) - 1)))))[:, 0]
#y_pred_rescaled = scaler.inverse_transform(np.column_stack((np.zeros((len(y_pred), 4)), y_pred)))[:, 4]

# Reescala los valores actuales (y_test) de la misma manera que las predicciones para poder compararlos en su escala original.
#y_actual_rescaled = scaler.inverse_transform(np.column_stack((y_test.reshape(-1, 1), np.zeros((len(y_test), len(features) - 1)))))[:, 0]
#y_actual_rescaled = scaler.inverse_transform(np.column_stack((np.zeros((len(y_test), 4)), y_test.reshape(-1, 1))))[:, 4]

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
y_pred_rescaled = get_original_scale(y_pred, scaler, len(features))
y_actual_rescaled = get_original_scale(y_test, scaler, len(features))

# Evaluación del modelo con métricas de error
rmse = np.sqrt(mean_squared_error(y_actual_rescaled, y_pred_rescaled))
mae = mean_absolute_error(y_actual_rescaled, y_pred_rescaled)
r2 = r2_score(y_actual_rescaled, y_pred_rescaled)

# Calculo del R^2 ajustado
n = len(y_actual_rescaled)  # Número de muestras
p = X_test.shape[1] # Número de predictores (en este caso, el número de sensores)
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R^2: {r2:.4f}")
print(f"R^2 ajustado: {r2_adj:.4f}")

# Representación gráfica de la pérdida de entrenamiento y validación a lo largo de las épocas. 
# Esto ayuda a visualizar cómo el modelo está aprendiendo y si hay sobreajuste o subajuste.
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_validation_loss.png', dpi=300, bbox_inches='tight')  # Guarda la figura como un archivo PNG
plt.show()

# Representación gráfica de las predicciones vs los valores reales.
plt.figure(figsize=(12, 6))
plt.plot(y_actual_rescaled, label='Actual Values')
plt.plot(y_pred_rescaled, label='Predicted Values')
plt.title('Predicted vs Actual Values')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.savefig('predicted_vs_actual.png', dpi=300, bbox_inches='tight')  # Guarda la figura como un archivo PNG
plt.show()
